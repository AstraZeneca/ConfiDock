"""Tests for the ESMFold2 msaPath baker.

The script exists because the ESMFold2 arm otherwise resolves its MSAs by hashing sequences at run time,
which makes the alignment a property of whichever directory the fold job mounted rather than of the input.
Baking the paths in only helps if two things hold, and both are what is asserted here:

* a chain with no ``.a3m`` comes out with *no* key rather than a path to a file that is not there, because
  a dangling ``msaPath`` makes ``resolve_chain_a3m`` raise where the old behaviour was to fold MSA-free
  with a warning -- turning a soft degradation into 8 dead GPU jobs;
* every path that *was* written resolves, checked by re-reading the files after the writes rather than
  trusting the writer, since this is the last point before the paths are handed to the fold jobs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflows" / "scripts" / "processing"))

import bake_esmfold2_msa_paths as bake_mod  # noqa: E402

KEY = bake_mod.MSA_PATH_KEY

HEAVY = "QVQLVQSGAEVKKPGASVKVSCKAS"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCRAS"


def _entry(name: str, *chains: dict) -> dict:
    return {"name": name, "sequences": list(chains)}


def _chain(chain_id: str, sequence: str, kind: str = "protein", **extra) -> dict:
    return {"id": chain_id, "type": kind, "sequence": sequence, **extra}


def _a3m(msa_dir: Path, sequence: str) -> Path:
    """Write the ``.a3m`` the runner's hash lookup would find for ``sequence``."""
    path = msa_dir / f"{hashlib.sha256(sequence.encode()).hexdigest()[:16]}.a3m"
    path.write_text(f">query\n{sequence}\n", encoding="utf-8")
    return path


@pytest.fixture
def msa_dir(tmp_path):
    directory = tmp_path / "sabdab_msas"
    directory.mkdir()
    return directory


class TestBakeEntry:
    """Tests for bake_entry."""

    def test_a_chain_with_an_alignment_gains_the_path(self, msa_dir):
        """The whole point: the input names the file instead of a hash lookup finding it later."""
        expected = _a3m(msa_dir, HEAVY)

        baked, found, missing = bake_mod.bake_entry(_entry("e", _chain("Hc", HEAVY)), msa_dir, {})

        assert baked["sequences"][0][KEY] == str(expected)
        assert (found, missing) == (1, 0)

    def test_a_chain_without_one_is_left_without_the_key(self, msa_dir):
        """Not a path to a missing file: ``resolve_chain_a3m`` raises on a dangling ``msaPath``, so writing
        one would turn "fold MSA-free and warn" into a hard failure at fold time."""
        baked, found, missing = bake_mod.bake_entry(_entry("e", _chain("Hc", HEAVY)), msa_dir, {})

        assert KEY not in baked["sequences"][0]
        assert (found, missing) == (0, 1)

    def test_a_stale_path_on_an_uncovered_chain_is_stripped(self, msa_dir):
        """Re-baking an already-baked set against a smaller MSA directory must not leave the old path
        behind: it would be verified against a file that has gone."""
        chain = _chain("Hc", HEAVY, **{KEY: "/gone/deadbeef.a3m"})

        baked, _, missing = bake_mod.bake_entry(_entry("e", chain), msa_dir, {})

        assert KEY not in baked["sequences"][0]
        assert missing == 1

    def test_non_protein_chains_are_passed_through_untouched(self, msa_dir):
        """A ligand has no sequence to hash, and is not counted either way."""
        ligand = {"id": "L1", "type": "ligand", "smiles": "CCO"}

        baked, found, missing = bake_mod.bake_entry(_entry("e", ligand), msa_dir, {})

        assert baked["sequences"] == [ligand]
        assert (found, missing) == (0, 0)

    def test_the_type_test_is_case_insensitive(self, msa_dir):
        """The inputs in this set carry both ``protein`` and ``Protein``; a case-sensitive test would
        silently skip every antigen chain."""
        _a3m(msa_dir, HEAVY)

        baked, found, _ = bake_mod.bake_entry(_entry("e", _chain("A", HEAVY, kind="Protein")), msa_dir, {})

        assert baked["sequences"][0][KEY]
        assert found == 1

    def test_the_input_entry_is_not_mutated(self, msa_dir):
        """Both the original and the baked entry are kept, for the coverage report."""
        _a3m(msa_dir, HEAVY)
        entry = _entry("e", _chain("Hc", HEAVY))

        bake_mod.bake_entry(entry, msa_dir, {})

        assert KEY not in entry["sequences"][0]

    def test_every_other_field_survives(self, msa_dir):
        """Only ``sequences`` is rebuilt; the entry name is what the fold job writes its outputs under."""
        entry = {"name": "1abc_x_2def", "extra": {"seed": 1}, "sequences": [_chain("Hc", HEAVY)]}

        baked, _, _ = bake_mod.bake_entry(entry, msa_dir, {})

        assert baked["name"] == "1abc_x_2def"
        assert baked["extra"] == {"seed": 1}

    def test_the_chain_order_is_preserved(self, msa_dir):
        """Chain order is the model's input order; reordering it changes the structure that comes out."""
        entry = _entry("e", _chain("Hc", HEAVY), _chain("Lc", LIGHT), {"id": "L1", "type": "ligand"})

        baked, _, _ = bake_mod.bake_entry(entry, msa_dir, {})

        assert [chain["id"] for chain in baked["sequences"]] == ["Hc", "Lc", "L1"]

    def test_a_repeated_sequence_is_resolved_once(self, msa_dir):
        """A combinatorial pairing set reuses chains -- the same heavy chain appears in every pairing --
        so the memo is what keeps this a few thousand lookups instead of a few hundred thousand."""
        _a3m(msa_dir, HEAVY)
        cache: dict[str, Path | None] = {}

        bake_mod.bake_entry(_entry("a", _chain("Hc", HEAVY)), msa_dir, cache)
        bake_mod.bake_entry(_entry("b", _chain("Hc", HEAVY)), msa_dir, cache)

        assert list(cache) == [HEAVY]

    def test_a_cached_miss_is_not_retried(self, msa_dir):
        """A None in the memo is a resolved answer, not an empty slot: ``if seq not in cache`` rather than
        ``if not cache.get(seq)``, or every uncovered sequence is re-hashed for every pairing it is in."""
        cache: dict[str, Path | None] = {HEAVY: None}
        _a3m(msa_dir, HEAVY)

        baked, found, missing = bake_mod.bake_entry(_entry("e", _chain("Hc", HEAVY)), msa_dir, cache)

        assert KEY not in baked["sequences"][0]
        assert (found, missing) == (0, 1)


class TestMsaPaths:
    """Tests for msa_paths."""

    def test_pairs_each_path_with_the_entry_that_carries_it(self):
        """A bare list of paths would say "some shard has a dangling msaPath" and leave the reader to find
        which of 9,013 entries it was."""
        entries = [_entry("1abc_native", _chain("Hc", HEAVY, **{KEY: "/msas/aaa.a3m"}))]

        assert bake_mod.msa_paths(entries) == [("1abc_native", "/msas/aaa.a3m")]

    def test_chains_without_the_key_are_absent(self):
        """No key is the intended state for a sequence with no alignment, not something to verify."""
        entries = [_entry("e", _chain("Hc", HEAVY), _chain("Lc", LIGHT, **{KEY: "/msas/bbb.a3m"}))]

        assert bake_mod.msa_paths(entries) == [("e", "/msas/bbb.a3m")]

    def test_a_shard_naming_no_alignments_gives_nothing(self):
        """Legitimate, if the MSA directory covers none of these sequences -- and not an error."""
        assert bake_mod.msa_paths([_entry("e", _chain("Hc", HEAVY))]) == []

    def test_file_order_is_preserved_across_entries_and_chains(self):
        """The first dangling path reported is then the first one in the file, which is where a reader
        looking at the JSON will start."""
        entries = [
            _entry("first", _chain("Hc", HEAVY, **{KEY: "/1.a3m"}), _chain("Lc", LIGHT, **{KEY: "/2.a3m"})),
            _entry("second", _chain("Hc", HEAVY, **{KEY: "/3.a3m"})),
        ]

        assert [path for _, path in bake_mod.msa_paths(entries)] == ["/1.a3m", "/2.a3m", "/3.a3m"]

    def test_a_blank_path_is_still_returned(self):
        """Tested on presence of the key, not on truthiness: an empty string is a broken path that the
        fold job would raise on, and it must reach the existence check rather than pass as "no MSA"."""
        entries = [_entry("e", _chain("Hc", HEAVY, **{KEY: ""}))]

        assert bake_mod.msa_paths(entries) == [("e", "")]

    def test_it_does_not_filter_by_chain_type(self):
        """Whatever names a path gets verified. The writer only ever puts one on a protein chain, so a key
        anywhere else means something upstream is wrong -- exactly the case not to skip."""
        entries = [_entry("e", {"id": "L1", "type": "ligand", KEY: "/1.a3m"})]

        assert bake_mod.msa_paths(entries) == [("e", "/1.a3m")]

    def test_an_unnamed_entry_still_reports_something(self):
        """``name`` is written by the input builder, but a hand-edited shard may not have one, and the
        verification must not die on the way to reporting the real problem."""
        entries = [{"sequences": [_chain("Hc", HEAVY, **{KEY: "/1.a3m"})]}]

        assert bake_mod.msa_paths(entries) == [("None", "/1.a3m")]


class TestVerifyShard:
    """Tests for verify_shard."""

    def test_returns_the_number_of_resolving_paths(self, msa_dir):
        """The count is what the log line quotes, so it has to be paths and not chains or entries."""
        first, second = _a3m(msa_dir, HEAVY), _a3m(msa_dir, LIGHT)
        entries = [_entry("e", _chain("Hc", HEAVY, **{KEY: str(first)}), _chain("Lc", LIGHT, **{KEY: str(second)}))]

        assert bake_mod.verify_shard("s.json", entries, 1) == 2

    def test_a_shard_naming_no_alignments_verifies_as_zero(self):
        """Not an error: every one of these sequences folds MSA-free, which the coverage report already
        said. Exiting here would block a legitimate run."""
        assert bake_mod.verify_shard("s.json", [_entry("e", _chain("Hc", HEAVY))], 1) == 0

    def test_a_dangling_path_is_fatal(self, msa_dir):
        """The whole reason to re-read the files: a path that is not there fails the fold job hours later
        and 8 times over."""
        entries = [_entry("e", _chain("Hc", HEAVY, **{KEY: str(msa_dir / "gone.a3m")}))]

        with pytest.raises(SystemExit):
            bake_mod.verify_shard("s.json", entries, 1)

    def test_the_dangling_path_message_names_the_shard_entry_and_path(self, msa_dir):
        """Enough to fix it without re-running anything: which file, which entry, which path."""
        missing = msa_dir / "gone.a3m"
        entries = [_entry("1abc_x_2def", _chain("Hc", HEAVY, **{KEY: str(missing)}))]

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("shard3.json", entries, 1)

        assert "shard3.json" in str(excinfo.value)
        assert "1abc_x_2def" in str(excinfo.value)
        assert str(missing) in str(excinfo.value)

    def test_a_blank_path_is_fatal(self):
        """``Path("").exists()`` is False, and a chain carrying an empty ``msaPath`` is as broken as one
        carrying a path to a deleted file."""
        with pytest.raises(SystemExit):
            bake_mod.verify_shard("s.json", [_entry("e", _chain("Hc", HEAVY, **{KEY: ""}))], 1)

    def test_the_first_dangling_path_is_the_one_reported(self, msa_dir):
        """Failing on the first keeps the message about one entry rather than a wall of them."""
        entries = [
            _entry("first", _chain("Hc", HEAVY, **{KEY: str(msa_dir / "a.a3m")})),
            _entry("second", _chain("Hc", HEAVY, **{KEY: str(msa_dir / "b.a3m")})),
        ]

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("s.json", entries, 2)

        assert "first" in str(excinfo.value)

    def test_a_short_shard_is_fatal(self):
        """A truncated write is silent otherwise: the fold grid would just be missing entries, and the
        gap looks like a set that was always smaller."""
        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("s.json", [_entry("e", _chain("Hc", HEAVY))], 2)

        assert "wrote 2 entries but re-read 1" in str(excinfo.value)

    def test_the_count_is_checked_before_the_paths(self, msa_dir):
        """A shard that is both short and dangling reports the count: it means the file itself is not what
        the writer thinks it is, which makes anything else read out of it suspect."""
        entries = [_entry("e", _chain("Hc", HEAVY, **{KEY: str(msa_dir / "gone.a3m")}))]

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("s.json", entries, 5)

        assert "re-read" in str(excinfo.value)
        assert "dangling" not in str(excinfo.value)

    def test_paths_are_counted_across_every_entry(self, msa_dir):
        """One shard holds many entries, and the log line is the shard's total."""
        path = _a3m(msa_dir, HEAVY)
        entries = [_entry(f"e{index}", _chain("Hc", HEAVY, **{KEY: str(path)})) for index in range(3)]

        assert bake_mod.verify_shard("s.json", entries, 3) == 3


class TestVerify:
    """Tests for verify."""

    def _write(self, out_dir: Path, name: str, entries: list[dict]) -> None:
        (out_dir / name).write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def test_every_expected_shard_is_read_back(self, tmp_path, msa_dir, caplog):
        """The manifest of what was written is the loop, so a shard that was never written is caught by
        the read rather than skipped."""
        path = _a3m(msa_dir, HEAVY)
        for name in ("s0.json", "s1.json"):
            self._write(tmp_path, name, [_entry("e", _chain("Hc", HEAVY, **{KEY: str(path)}))])

        with caplog.at_level(logging.INFO):
            bake_mod.verify(tmp_path, {"s0.json": 1, "s1.json": 1})

        assert [record.getMessage() for record in caplog.records] == [
            "verified s0.json: 1 entries, 1 msaPath values all resolve",
            "verified s1.json: 1 entries, 1 msaPath values all resolve",
        ]

    def test_the_logged_count_is_the_shards_own(self, tmp_path, msa_dir, caplog):
        """Per shard, not cumulative: the numbers are read to spot the shard that is different."""
        path = _a3m(msa_dir, HEAVY)
        self._write(tmp_path, "s0.json", [_entry("e", _chain("Hc", HEAVY, **{KEY: str(path)}))])
        self._write(tmp_path, "s1.json", [_entry("e", _chain("Hc", HEAVY))])

        with caplog.at_level(logging.INFO):
            bake_mod.verify(tmp_path, {"s0.json": 1, "s1.json": 1})

        assert "1 msaPath values" in caplog.records[0].getMessage()
        assert "0 msaPath values" in caplog.records[1].getMessage()

    def test_a_bad_shard_stops_the_run(self, tmp_path, msa_dir):
        """It exits rather than carrying on to the remaining shards: the set is folded as a whole."""
        self._write(tmp_path, "s0.json", [_entry("e", _chain("Hc", HEAVY, **{KEY: str(msa_dir / "gone.a3m")}))])

        with pytest.raises(SystemExit):
            bake_mod.verify(tmp_path, {"s0.json": 1})

    def test_a_shard_that_was_never_written_raises(self, tmp_path):
        """Rather than an empty verification that reports success over nothing."""
        with pytest.raises(FileNotFoundError):
            bake_mod.verify(tmp_path, {"missing.json": 1})


class TestReportCoverage:
    """Tests for report_coverage."""

    def test_the_two_chain_kinds_are_counted_separately(self, caplog):
        """The interesting number is antigen coverage: a newly paired antigen is a chain the MSA directory
        was never built for, and pooling it with the binder's hides that."""
        baked = [
            _entry(
                "e",
                _chain("Hc", HEAVY, **{KEY: "/1.a3m"}),
                _chain("A", "MKT", kind="Protein"),
            )
        ]

        with caplog.at_level(logging.INFO):
            bake_mod.report_coverage(baked, baked)

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "antibody chains: 1/1" in messages
        assert "antigen chains: 0/1" in messages

    def test_chains_are_split_on_the_h_l_prefix(self, caplog):
        """Mirrors the runner's own fallback, so the report describes what the fold job will do."""
        baked = [_entry("e", _chain("L", LIGHT), _chain("B", "MKT"))]

        with caplog.at_level(logging.INFO):
            bake_mod.report_coverage(baked, baked)

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "antibody chains: 0/1" in messages
        assert "antigen chains: 0/1" in messages

    def test_non_protein_chains_are_not_counted(self, caplog):
        """A ligand has no alignment to be missing, and counting it would understate coverage."""
        baked = [_entry("e", _chain("Hc", HEAVY, **{KEY: "/1.a3m"}), {"id": "L1", "type": "ligand"})]

        with caplog.at_level(logging.INFO):
            bake_mod.report_coverage(baked, baked)

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "antibody chains: 1/1" in messages
        assert "antigen" not in messages

    def test_the_entry_counts_are_logged_from_both_sides(self, caplog):
        """In and out are counted separately so a shard that lost an entry in baking shows up here."""
        original = [_entry("e", _chain("Hc", HEAVY))]

        with caplog.at_level(logging.INFO):
            bake_mod.report_coverage(original + original, original)

        assert "entries: 2 in, 1 out" in caplog.records[-1].getMessage()
