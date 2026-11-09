"""Tests for confidock.protenix.msa module."""

from __future__ import annotations

import hashlib

from confidock.protenix.msa import (
    inject_msa,
    inject_msa_paths,
    resolve_a3m,
    stage_msa_dirs,
    strip_msa,
)


class TestResolveA3m:
    """Tests for resolve_a3m function."""

    def test_resolve_existing_a3m(self, tmp_path):
        """Test resolving an existing .a3m file."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        seq = "ACDEFGHIKLMNPQRSTVWY"
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        a3m_file = msa_dir / f"{digest}.a3m"
        a3m_file.write_text(">query\n" + seq + "\n")

        result = resolve_a3m(seq, msa_dir)

        assert result == a3m_file
        assert result.exists()

    def test_resolve_nonexistent_a3m(self, tmp_path):
        """Test resolving a non-existent .a3m file."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        seq = "ACDEFGHIKLMNPQRSTVWY"

        result = resolve_a3m(seq, msa_dir)

        assert result is None

    def test_resolve_empty_a3m(self, tmp_path):
        """Test resolving an empty .a3m file."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        seq = "ACDEFGHIKLMNPQRSTVWY"
        digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
        a3m_file = msa_dir / f"{digest}.a3m"
        a3m_file.write_text("")  # Empty file

        result = resolve_a3m(seq, msa_dir)

        assert result is None


class TestStageMsaDirs:
    """Tests for stage_msa_dirs function."""

    def test_stage_with_existing_msas(self, tmp_path):
        """Test staging MSA directories with existing .a3m files."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        staging_dir = tmp_path / "staging"

        # Create test entries
        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "MKLPQRSTVWYACDEFGHIK"

        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"proteinChain": {"sequence": seq1}},
                    {"proteinChain": {"sequence": seq2}},
                ],
            }
        ]

        # Create .a3m files
        for seq in [seq1, seq2]:
            digest = hashlib.sha256(seq.encode()).hexdigest()[:16]
            a3m_file = msa_dir / f"{digest}.a3m"
            a3m_file.write_text(f">query\n{seq}\n>hit1\nAAAAAAAAAAAAAAAAAAAA\n")

        result = stage_msa_dirs(entries, msa_dir, staging_dir)

        assert len(result) == 2
        assert seq1 in result
        assert seq2 in result

        # Check that staging directories were created
        for seq in [seq1, seq2]:
            seq_dir = result[seq]
            assert seq_dir.exists()
            assert (seq_dir / "non_pairing.a3m").exists()
            assert (seq_dir / "pairing.a3m").exists()

    def test_stage_with_missing_msas(self, tmp_path):
        """Test staging MSA directories when .a3m files are missing."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        staging_dir = tmp_path / "staging"

        seq = "ACDEFGHIKLMNPQRSTVWY"
        entries = [
            {
                "name": "entry1",
                "sequences": [{"proteinChain": {"sequence": seq}}],
            }
        ]

        result = stage_msa_dirs(entries, msa_dir, staging_dir)

        assert len(result) == 1
        assert seq in result

        # Check that query-only MSA was written
        seq_dir = result[seq]
        non_pairing = seq_dir / "non_pairing.a3m"
        assert non_pairing.exists()
        content = non_pairing.read_text()
        assert ">query" in content
        assert seq in content

    def test_stage_deduplicates_sequences(self, tmp_path):
        """Test that duplicate sequences are handled correctly."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        staging_dir = tmp_path / "staging"

        seq = "ACDEFGHIKLMNPQRSTVWY"

        # Same sequence appears twice
        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"proteinChain": {"sequence": seq}},
                    {"proteinChain": {"sequence": seq}},  # Duplicate
                ],
            }
        ]

        result = stage_msa_dirs(entries, msa_dir, staging_dir)

        # Should only create one staging directory
        assert len(result) == 1

    def test_stage_skips_non_protein_chains(self, tmp_path):
        """Test that non-protein chains are skipped."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()
        staging_dir = tmp_path / "staging"

        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"dnaChain": {"sequence": "ATCGATCG"}},  # DNA, not protein
                ],
            }
        ]

        result = stage_msa_dirs(entries, msa_dir, staging_dir)

        assert len(result) == 0


class TestInjectMsa:
    """Tests for inject_msa function."""

    def test_inject_msa_paths(self, tmp_path):
        """Test injecting MSA paths into entries."""
        msa_dir1 = tmp_path / "msa1"
        msa_dir2 = tmp_path / "msa2"
        msa_dir1.mkdir()
        msa_dir2.mkdir()

        seq1 = "ACDEFGHIKLMNPQRSTVWY"
        seq2 = "MKLPQRSTVWYACDEFGHIK"

        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"proteinChain": {"sequence": seq1}},
                    {"proteinChain": {"sequence": seq2}},
                ],
            }
        ]

        seq_to_msa_dir = {seq1: msa_dir1, seq2: msa_dir2}

        result = inject_msa(entries, seq_to_msa_dir)

        assert len(result) == 1
        assert result[0]["name"] == "entry1"
        assert len(result[0]["sequences"]) == 2

        # Check MSA was injected
        pc1 = result[0]["sequences"][0]["proteinChain"]
        assert "msa" in pc1
        assert pc1["msa"]["precomputed_msa_dir"] == str(msa_dir1)
        assert pc1["msa"]["pairing_db"] == "uniref100"

        pc2 = result[0]["sequences"][1]["proteinChain"]
        assert "msa" in pc2
        assert pc2["msa"]["precomputed_msa_dir"] == str(msa_dir2)

    def test_inject_msa_skips_non_protein(self, tmp_path):
        """Test that non-protein chains are handled correctly."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"dnaChain": {"sequence": "ATCGATCG"}},
                ],
            }
        ]

        seq_to_msa_dir = {}

        result = inject_msa(entries, seq_to_msa_dir)

        # Should not crash, should return entry unchanged
        assert len(result) == 1
        assert "dnaChain" in result[0]["sequences"][0]

    def test_inject_msa_deep_copies(self, tmp_path):
        """Test that inject_msa makes deep copies."""
        msa_dir = tmp_path / "msa"
        msa_dir.mkdir()

        seq = "ACDEFGHIKLMNPQRSTVWY"
        entries = [
            {
                "name": "entry1",
                "sequences": [{"proteinChain": {"sequence": seq}}],
            }
        ]

        seq_to_msa_dir = {seq: msa_dir}

        result = inject_msa(entries, seq_to_msa_dir)

        # Modify result
        result[0]["name"] = "modified"

        # Original should be unchanged
        assert entries[0]["name"] == "entry1"


class TestInjectMsaPaths:
    """Tests for inject_msa_paths function (Protenix's current MSA schema)."""

    @staticmethod
    def _staged(tmp_path, name, *, pairing=True, non_pairing=True):
        """Build a staged per-sequence MSA directory."""
        d = tmp_path / name
        d.mkdir()
        if pairing:
            (d / "pairing.a3m").write_text(">query\nACDEFG\n")
        if non_pairing:
            (d / "non_pairing.a3m").write_text(">query\nACDEFG\n>hit1\nGGGGGG\n")
        return d

    def test_writes_both_path_keys(self, tmp_path):
        """Both pairedMsaPath and unpairedMsaPath point at the staged files."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        staged = self._staged(tmp_path, "sha1")
        entries = [{"name": "entry1", "sequences": [{"proteinChain": {"sequence": seq}}]}]

        result = inject_msa_paths(entries, {seq: staged})

        pc = result[0]["sequences"][0]["proteinChain"]
        assert pc["pairedMsaPath"] == str(staged / "pairing.a3m")
        assert pc["unpairedMsaPath"] == str(staged / "non_pairing.a3m")

    def test_omits_missing_files(self, tmp_path):
        """A path is only written when the file exists, so Protenix never sees a dangling one."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        staged = self._staged(tmp_path, "sha1", pairing=False)
        entries = [{"name": "entry1", "sequences": [{"proteinChain": {"sequence": seq}}]}]

        result = inject_msa_paths(entries, {seq: staged})

        pc = result[0]["sequences"][0]["proteinChain"]
        assert "pairedMsaPath" not in pc
        assert pc["unpairedMsaPath"] == str(staged / "non_pairing.a3m")

    def test_drops_legacy_msa_block(self, tmp_path):
        """The legacy msa block is removed so the two schemas cannot disagree."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        staged = self._staged(tmp_path, "sha1")
        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"proteinChain": {"sequence": seq, "msa": {"precomputed_msa_dir": "/stale"}}},
                ],
            }
        ]

        result = inject_msa_paths(entries, {seq: staged})

        assert "msa" not in result[0]["sequences"][0]["proteinChain"]

    def test_include_paired_false_writes_only_unpaired(self, tmp_path):
        """include_paired=False references the unpaired MSA alone."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        staged = self._staged(tmp_path, "sha1")
        entries = [{"name": "entry1", "sequences": [{"proteinChain": {"sequence": seq}}]}]

        result = inject_msa_paths(entries, {seq: staged}, include_paired=False)

        pc = result[0]["sequences"][0]["proteinChain"]
        assert "pairedMsaPath" not in pc
        assert pc["unpairedMsaPath"] == str(staged / "non_pairing.a3m")

    def test_include_paired_false_drops_inherited_paired_path(self, tmp_path):
        """A pairedMsaPath already on the input is removed, not left stale."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        staged = self._staged(tmp_path, "sha1")
        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {"proteinChain": {"sequence": seq, "pairedMsaPath": "/stale/pairing.a3m"}},
                ],
            }
        ]

        result = inject_msa_paths(entries, {seq: staged}, include_paired=False)

        assert "pairedMsaPath" not in result[0]["sequences"][0]["proteinChain"]

    def test_unknown_sequence_left_alone(self):
        """A sequence absent from the mapping is passed through untouched."""
        entries = [{"name": "entry1", "sequences": [{"proteinChain": {"sequence": "ACDEFG"}}]}]

        result = inject_msa_paths(entries, {})

        assert result[0]["sequences"][0]["proteinChain"] == {"sequence": "ACDEFG"}

    def test_skips_non_protein(self):
        """Non-protein chains do not crash the injector."""
        entries = [{"name": "entry1", "sequences": [{"dnaChain": {"sequence": "ATCGATCG"}}]}]

        result = inject_msa_paths(entries, {})

        assert "dnaChain" in result[0]["sequences"][0]

    def test_deep_copies(self, tmp_path):
        """The input entries are not mutated."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        staged = self._staged(tmp_path, "sha1")
        entries = [{"name": "entry1", "sequences": [{"proteinChain": {"sequence": seq}}]}]

        inject_msa_paths(entries, {seq: staged})

        assert entries[0]["sequences"][0]["proteinChain"] == {"sequence": seq}


class TestStripMsa:
    """Tests for strip_msa function."""

    def test_strip_msa_removes_new_format_paths(self):
        """A no_msa condition stays MSA-free even when paths are baked into the input JSON."""
        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "ACDEFG",
                            "pairedMsaPath": "/staged/sha1/pairing.a3m",
                            "unpairedMsaPath": "/staged/sha1/non_pairing.a3m",
                        }
                    }
                ],
            }
        ]

        result = strip_msa(entries)

        assert result[0]["sequences"][0]["proteinChain"] == {"sequence": "ACDEFG"}

    def test_strip_msa_removes_msa_field(self):
        """Test that strip_msa removes msa fields."""
        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "ACDEFG",
                            "msa": {"precomputed_msa_dir": "/path/to/msa"},
                        }
                    }
                ],
            }
        ]

        result = strip_msa(entries)

        assert len(result) == 1
        pc = result[0]["sequences"][0]["proteinChain"]
        assert "msa" not in pc
        assert "sequence" in pc

    def test_strip_msa_handles_missing_msa(self):
        """Test that strip_msa handles entries without msa."""
        entries = [
            {
                "name": "entry1",
                "sequences": [{"proteinChain": {"sequence": "ACDEFG"}}],
            }
        ]

        result = strip_msa(entries)

        assert len(result) == 1
        assert "proteinChain" in result[0]["sequences"][0]

    def test_strip_msa_deep_copies(self):
        """Test that strip_msa makes deep copies."""
        entries = [
            {
                "name": "entry1",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": "ACDEFG",
                            "msa": {"precomputed_msa_dir": "/path/to/msa"},
                        }
                    }
                ],
            }
        ]

        result = strip_msa(entries)

        # Modify result
        result[0]["name"] = "modified"

        # Original should be unchanged
        assert entries[0]["name"] == "entry1"
