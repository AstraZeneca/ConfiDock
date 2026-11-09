"""Tests for the verification half of the Protenix msaPath baker.

Protenix reads the baked JSON with ``--use_msa true`` and no ``--msa_dir``, which makes the paths in the
file the only thing standing between the run and a live MMseqs2 search: ``need_msa_search`` treats a
*missing* path as a reason to go and search. So a wrong path here does not crash the fold -- it silently
turns a 3-minute staged run into an hours-long search over the network, on 8 GPUs at once. That is why the
written shards are re-read and checked rather than trusted, and it is what every test below is about.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflows" / "scripts" / "processing"))

import bake_protenix_msa_paths as bake_mod  # noqa: E402

PAIRED, UNPAIRED = "pairedMsaPath", "unpairedMsaPath"
BOTH = (PAIRED, UNPAIRED)

HEAVY = "QVQLVQSGAEVKKPGASVKVSCKAS"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCRAS"


def _protein(sequence: str, **msa_paths) -> dict:
    return {"proteinChain": {"sequence": sequence, "count": 1, **msa_paths}}


def _entry(name: str, *chains: dict) -> dict:
    return {"name": name, "sequences": list(chains)}


@pytest.fixture
def staged(tmp_path):
    """A staging directory holding both files ``stage_msa_dirs`` writes per sequence."""
    directory = tmp_path / "protenix_msa_staging" / "abcdef0123456789"
    directory.mkdir(parents=True)
    (directory / "pairing.a3m").write_text(f">query\n{HEAVY}\n", encoding="utf-8")
    (directory / "non_pairing.a3m").write_text(f">query\n{HEAVY}\n", encoding="utf-8")
    return directory


def _both_paths(staged: Path) -> dict:
    return {PAIRED: str(staged / "pairing.a3m"), UNPAIRED: str(staged / "non_pairing.a3m")}


class TestProteinChains:
    """Tests for protein_chains."""

    def test_pairs_each_chain_with_the_entry_that_carries_it(self):
        """A bare list of chain bodies would say "some shard has a dangling path" and leave the reader to
        find which of 9,013 entries it was."""
        chain = _protein(HEAVY, **{PAIRED: "/1.a3m"})

        assert bake_mod.protein_chains([_entry("1abc_native", chain)]) == [
            ("1abc_native", chain["proteinChain"]),
        ]

    def test_non_protein_chains_are_absent(self):
        """A ligand has no sequence to align and Protenix asks for no MSA key on it, so verifying one
        would fail every entry that has one."""
        entries = [_entry("e", {"ligand": {"ligand": "CCD_ATP", "count": 1}}, _protein(HEAVY))]

        assert [name for name, _ in bake_mod.protein_chains(entries)] == ["e"]
        assert len(bake_mod.protein_chains(entries)) == 1

    def test_the_chain_body_is_returned_not_the_wrapper(self):
        """The MSA keys live inside ``proteinChain``, so returning the wrapper would look for them one
        level too high and report every chain as missing them."""
        entries = [_entry("e", _protein(HEAVY, **{PAIRED: "/1.a3m"}))]

        _, chain = bake_mod.protein_chains(entries)[0]

        assert chain[PAIRED] == "/1.a3m"

    def test_file_order_is_preserved_across_entries_and_chains(self):
        """The first bad chain reported is then the first one in the file, which is where a reader looking
        at the JSON will start."""
        entries = [
            _entry("first", _protein(HEAVY, **{PAIRED: "/1"}), _protein(LIGHT, **{PAIRED: "/2"})),
            _entry("second", _protein(HEAVY, **{PAIRED: "/3"})),
        ]

        assert [chain[PAIRED] for _, chain in bake_mod.protein_chains(entries)] == ["/1", "/2", "/3"]

    def test_an_entry_with_no_protein_chains_contributes_nothing(self):
        """Not an error here -- if it is wrong, the entry count check is what catches it."""
        assert bake_mod.protein_chains([_entry("e", {"ligand": {"ligand": "CCD_ATP"}})]) == []

    def test_a_chain_carrying_no_msa_keys_is_still_returned(self):
        """It is exactly the chain that must be reported: filtering on the keys' presence would make an
        unbaked chain invisible to the check meant to find it."""
        entries = [_entry("e", _protein(HEAVY))]

        assert bake_mod.protein_chains(entries) == [("e", {"sequence": HEAVY, "count": 1})]

    def test_an_unnamed_entry_still_reports_something(self):
        """``name`` is written by the input builder, but a hand-edited shard may not have one, and the
        verification must not die on the way to reporting the real problem."""
        entries = [{"sequences": [_protein(HEAVY)]}]

        assert bake_mod.protein_chains(entries)[0][0] == "None"


class TestVerifyChain:
    """Tests for verify_chain."""

    def test_a_fully_staged_chain_passes(self, staged):
        """The normal case: both files are where the JSON says they are."""
        bake_mod.verify_chain("s.json", "e", _both_paths(staged), BOTH)

    def test_a_missing_key_is_fatal(self, staged):
        """This is the script failing to bake, and Protenix would read the absent key as "go and search"."""
        chain = {UNPAIRED: str(staged / "non_pairing.a3m")}

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_chain("s.json", "1abc_native", chain, BOTH)

        assert "missing" in str(excinfo.value)
        assert PAIRED in str(excinfo.value)

    def test_a_dangling_path_is_fatal(self, staged):
        """The staging tree not being where the JSON says -- a different failure from a missing key, and
        reported differently so the two are not confused."""
        chain = {**_both_paths(staged), UNPAIRED: str(staged / "gone.a3m")}

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_chain("s.json", "1abc_native", chain, BOTH)

        assert "dangling" in str(excinfo.value)

    def test_the_message_names_the_shard_entry_and_key(self, staged):
        """Enough to fix it without re-running anything."""
        chain = {**_both_paths(staged), UNPAIRED: str(staged / "gone.a3m")}

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_chain("shard3.json", "1abc_x_2def", chain, BOTH)

        message = str(excinfo.value)
        assert "shard3.json" in message
        assert "1abc_x_2def" in message
        assert UNPAIRED in message

    def test_a_blank_path_is_fatal(self, staged):
        """``Path("").exists()`` is True -- it is ``Path(".")`` -- so a blank path would verify clean here
        and then send Protenix to read a directory as an alignment."""
        chain = {**_both_paths(staged), UNPAIRED: ""}

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_chain("s.json", "e", chain, BOTH)

        assert "dangling" in str(excinfo.value)

    def test_every_required_key_is_checked_not_just_the_first(self, staged):
        """The second key is the one that carries the real alignment; only checking the first would pass a
        chain whose actual MSA is missing."""
        chain = {PAIRED: str(staged / "pairing.a3m")}

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_chain("s.json", "e", chain, BOTH)

        assert UNPAIRED in str(excinfo.value)

    def test_only_the_required_keys_are_checked(self, staged):
        """``--no-pairing`` writes the unpaired path alone, on purpose: the staged ``pairing.a3m`` is a
        query-only dummy, so requiring it would fail a run that deliberately left it out."""
        chain = {UNPAIRED: str(staged / "non_pairing.a3m")}

        bake_mod.verify_chain("s.json", "e", chain, (UNPAIRED,))


class TestVerifyShard:
    """Tests for verify_shard."""

    def test_a_fully_staged_shard_passes(self, staged):
        """Silence is the pass: nothing is returned, and anything wrong exits."""
        entries = [_entry("e", _protein(HEAVY, **_both_paths(staged)))]

        assert bake_mod.verify_shard("s.json", entries, 1, BOTH) is None

    def test_a_short_shard_is_fatal(self):
        """A truncated write is silent otherwise: the fold grid would just be missing entries, and the gap
        looks like a set that was always smaller."""
        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("s.json", [_entry("e", _protein(HEAVY))], 2, BOTH)

        assert "wrote 2 entries but re-read 1" in str(excinfo.value)

    def test_the_count_is_checked_before_the_chains(self):
        """A shard that is both short and unbaked reports the count: it means the file itself is not what
        the writer thinks it is, which makes anything else read out of it suspect."""
        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("s.json", [_entry("e", _protein(HEAVY))], 5, BOTH)

        assert "re-read" in str(excinfo.value)
        assert "missing" not in str(excinfo.value)

    def test_the_last_chain_of_the_last_entry_is_still_checked(self, staged):
        """Every chain, not a sample: one unbaked chain out of thousands is enough to start a live MSA
        search on the job that reaches it."""
        good = _protein(HEAVY, **_both_paths(staged))
        entries = [_entry("first", good, good), _entry("second", good, _protein(LIGHT))]

        with pytest.raises(SystemExit) as excinfo:
            bake_mod.verify_shard("s.json", entries, 2, BOTH)

        assert "second" in str(excinfo.value)

    def test_a_ligand_chain_does_not_trip_it(self, staged):
        """Entries in this set carry ligands, and none of them can have an MSA."""
        entries = [_entry("e", _protein(HEAVY, **_both_paths(staged)), {"ligand": {"ligand": "CCD_ATP"}})]

        assert bake_mod.verify_shard("s.json", entries, 1, BOTH) is None

    def test_an_entry_count_of_zero_matches_an_empty_shard(self):
        """A shard can legitimately end up empty when the set is sharded more ways than it has entries."""
        assert bake_mod.verify_shard("s.json", [], 0, BOTH) is None


class TestVerify:
    """Tests for verify."""

    def _write(self, out_dir: Path, name: str, entries: list[dict]) -> None:
        (out_dir / name).write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def test_every_expected_shard_is_read_back(self, tmp_path, staged, caplog):
        """The manifest of what was written is the loop, so a shard that was never written is caught by the
        read rather than skipped."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        for name in ("s0.json", "s1.json"):
            self._write(out_dir, name, [_entry("e", _protein(HEAVY, **_both_paths(staged)))])

        with caplog.at_level(logging.INFO):
            bake_mod.verify(out_dir, {"s0.json": 1, "s1.json": 1}, BOTH)

        assert [record.getMessage() for record in caplog.records] == [
            "verified s0.json: 1 entries, all chains resolve",
            "verified s1.json: 1 entries, all chains resolve",
        ]

    def test_a_bad_shard_stops_the_run(self, tmp_path, staged):
        """It exits rather than carrying on to the remaining shards: the set is folded as a whole."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        self._write(out_dir, "s0.json", [_entry("e", _protein(HEAVY))])

        with pytest.raises(SystemExit):
            bake_mod.verify(out_dir, {"s0.json": 1}, BOTH)

    def test_nothing_is_logged_for_a_shard_that_failed(self, tmp_path, caplog):
        """ "verified" must mean verified: a log line emitted before the check would be read as a pass."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        self._write(out_dir, "s0.json", [_entry("e", _protein(HEAVY))])

        with caplog.at_level(logging.INFO), pytest.raises(SystemExit):
            bake_mod.verify(out_dir, {"s0.json": 1}, BOTH)

        assert not [record for record in caplog.records if "verified" in record.getMessage()]

    def test_the_required_keys_are_passed_through(self, tmp_path, staged, caplog):
        """A ``--no-pairing`` run writes one key per chain, and the same verification has to accept it."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        chain = _protein(HEAVY, **{UNPAIRED: str(staged / "non_pairing.a3m")})
        self._write(out_dir, "s0.json", [_entry("e", chain)])

        with caplog.at_level(logging.INFO):
            bake_mod.verify(out_dir, {"s0.json": 1}, (UNPAIRED,))

        assert "verified s0.json" in caplog.records[0].getMessage()

    def test_a_shard_that_was_never_written_raises(self, tmp_path):
        """Rather than an empty verification that reports success over nothing."""
        with pytest.raises(FileNotFoundError):
            bake_mod.verify(tmp_path, {"missing.json": 1}, BOTH)
