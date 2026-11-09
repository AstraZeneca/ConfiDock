"""Tests for confidock.evaluation.sabdab module."""

from confidock.evaluation.sabdab import (
    infer_chain_roles,
    interface_role_pair,
    load_sabdab_chain_types,
    load_sabdab_role_counts,
    role_sequence,
    split_chain_tokens,
)


class TestSplitChainTokens:
    """Tests for split_chain_tokens function."""

    def test_split_chain_tokens_basic(self):
        """Test basic pipe-delimited parsing."""
        result = split_chain_tokens("H|L")
        assert result == ["H", "L"]

    def test_split_chain_tokens_multiple(self):
        """Test parsing multiple chains."""
        result = split_chain_tokens("A|B|C")
        assert result == ["A", "B", "C"]

    def test_split_chain_tokens_with_spaces(self):
        """Test parsing with whitespace."""
        result = split_chain_tokens(" H | L ")
        assert result == ["H", "L"]

    def test_split_chain_tokens_with_na(self):
        """Test filtering out NA values."""
        result = split_chain_tokens("H|NA|L")
        assert result == ["H", "L"]

    def test_split_chain_tokens_na_only(self):
        """Test NA-only input."""
        result = split_chain_tokens("NA")
        assert result == []

    def test_split_chain_tokens_empty(self):
        """Test empty string."""
        result = split_chain_tokens("")
        assert result == []

    def test_split_chain_tokens_mixed_case_na(self):
        """Test case-insensitive NA filtering."""
        result = split_chain_tokens("H|na|L|Na|NA")
        assert result == ["H", "L"]


class TestRoleSequence:
    """Tests for role_sequence function."""

    def test_role_sequence_basic(self):
        """Test basic binder/antigen split."""
        result = role_sequence(n_binder=2, n_antigen=1, n_chains=3)
        assert result == ["binder", "binder", "antigen"]

    def test_role_sequence_padding_needed(self):
        """Test padding with extra antigen chains."""
        result = role_sequence(n_binder=1, n_antigen=1, n_chains=4)
        assert result == ["binder", "antigen", "antigen", "antigen"]

    def test_role_sequence_default_when_empty(self):
        """Test default (1 binder + rest antigen) when no roles specified."""
        result = role_sequence(n_binder=0, n_antigen=0, n_chains=3)
        assert result == ["binder", "antigen", "antigen"]

    def test_role_sequence_single_chain(self):
        """Test single chain case."""
        result = role_sequence(n_binder=1, n_antigen=0, n_chains=1)
        assert result == ["binder"]

    def test_role_sequence_truncate_if_oversized(self):
        """Test truncation if specified roles exceed n_chains."""
        result = role_sequence(n_binder=2, n_antigen=2, n_chains=3)
        assert result == ["binder", "binder", "antigen"]
        assert len(result) == 3

    def test_role_sequence_negative_binder(self):
        """Test handling of negative binder count."""
        result = role_sequence(n_binder=-1, n_antigen=2, n_chains=3)
        # max(n_binder, 0) should clamp to 0
        assert result == ["antigen", "antigen", "antigen"]

    def test_role_sequence_zero_chains(self):
        """Test zero chains."""
        result = role_sequence(n_binder=1, n_antigen=1, n_chains=0)
        assert result == []


class TestLoadSabdabRoleCounts:
    """Tests for load_sabdab_role_counts function."""

    def test_load_sabdab_role_counts_basic(self, tmp_path):
        """Test loading role counts from TSV."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tH\tL\tA|B\n2def\tH\tNA\tC\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        result = load_sabdab_role_counts(tsv_path)
        assert result == {
            "1abc": (2, 2),  # H + L = 2 binder, A|B = 2 antigen
            "2def": (1, 1),  # H only = 1 binder, C = 1 antigen
        }

    def test_load_sabdab_role_counts_missing_file(self, tmp_path):
        """Test with non-existent file."""
        result = load_sabdab_role_counts(tmp_path / "nonexistent.tsv")
        assert result == {}

    def test_load_sabdab_role_counts_none_path(self):
        """Test with None path."""
        result = load_sabdab_role_counts(None)
        assert result == {}

    def test_load_sabdab_role_counts_duplicate_pdb(self, tmp_path):
        """Test that duplicate PDB IDs are skipped (first entry kept)."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tH\tL\tA\n1abc\tH\tNA\tB|C\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        result = load_sabdab_role_counts(tsv_path)
        assert result == {"1abc": (2, 1)}  # First entry only

    def test_load_sabdab_role_counts_case_normalization(self, tmp_path):
        """Test PDB ID case normalization."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1ABC\tH\tL\tA\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        result = load_sabdab_role_counts(tsv_path)
        assert "1abc" in result  # Lowercased
        assert result["1abc"] == (2, 1)

    def test_load_sabdab_role_counts_empty_fields(self, tmp_path):
        """Test handling of empty/NA fields."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tNA\tNA\tNA\n2def\t\t\t\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        result = load_sabdab_role_counts(tsv_path)
        assert result == {
            "1abc": (0, 0),
            "2def": (0, 0),
        }


class TestLoadSabdabChainTypes:
    """Tests for load_sabdab_chain_types function."""

    def test_load_sabdab_chain_types_basic(self, tmp_path):
        """Test loading chain types from TSV."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tH\tL\tA|B\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        result = load_sabdab_chain_types(tsv_path)
        assert result == {
            "1abc": {
                "H": "heavy",
                "L": "light",
                "A": "antigen",
                "B": "antigen",
            }
        }

    def test_load_sabdab_chain_types_multiple_entries(self, tmp_path):
        """Test multiple PDB entries."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tH\tL\tA\n2def\tH|K\tNA\tB|C\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        result = load_sabdab_chain_types(tsv_path)
        assert result["1abc"] == {"H": "heavy", "L": "light", "A": "antigen"}
        assert result["2def"] == {"H": "heavy", "K": "heavy", "B": "antigen", "C": "antigen"}

    def test_load_sabdab_chain_types_missing_file(self, tmp_path):
        """Test with non-existent file."""
        result = load_sabdab_chain_types(tmp_path / "nonexistent.tsv")
        assert result == {}

    def test_load_sabdab_chain_types_none_path(self):
        """Test with None path."""
        result = load_sabdab_chain_types(None)
        assert result == {}


class TestInferChainRoles:
    """Tests for infer_chain_roles function."""

    def test_infer_chain_roles_basic(self):
        """Test basic role inference."""
        role_counts = {"1abc": (2, 1)}
        result = infer_chain_roles(["A", "B", "C"], "1abc", role_counts)
        assert result == {
            "A": "binder",
            "B": "binder",
            "C": "antigen",
        }

    def test_infer_chain_roles_unknown_pdb(self):
        """Test fallback for unknown PDB ID."""
        role_counts = {}
        result = infer_chain_roles(["A", "B", "C"], "unknown", role_counts)
        # Default: 1 binder + rest antigen for 3+ chains
        assert result == {
            "A": "binder",
            "B": "antigen",
            "C": "antigen",
        }

    def test_infer_chain_roles_single_chain(self):
        """Test single chain case."""
        role_counts = {}
        result = infer_chain_roles(["A"], "unknown", role_counts)
        # Default for single chain: 0 binder, 0 antigen → defaults to 1 binder
        assert result == {"A": "binder"}

    def test_infer_chain_roles_case_normalization(self):
        """Test PDB ID case normalization in lookup."""
        role_counts = {"1abc": (1, 2)}
        result = infer_chain_roles(["A", "B", "C"], "1ABC", role_counts)
        assert result == {
            "A": "binder",
            "B": "antigen",
            "C": "antigen",
        }


class TestInterfaceRolePair:
    """Tests for interface_role_pair function."""

    def test_interface_role_pair_binder_antigen(self):
        """Test binder-antigen interface."""
        chain_roles = {"A": "binder", "B": "antigen"}
        result = interface_role_pair("AB", chain_roles)
        assert result == "binder-antigen"

    def test_interface_role_pair_binder_binder(self):
        """Test binder-binder interface."""
        chain_roles = {"A": "binder", "B": "binder"}
        result = interface_role_pair("AB", chain_roles)
        assert result == "binder-binder"

    def test_interface_role_pair_antigen_antigen(self):
        """Test antigen-antigen interface."""
        chain_roles = {"A": "antigen", "B": "antigen"}
        result = interface_role_pair("AB", chain_roles)
        assert result == "antigen-antigen"

    def test_interface_role_pair_unknown_chain(self):
        """Test with unmapped chain (should warn and return unknown-unknown)."""
        chain_roles = {"A": "binder"}
        result = interface_role_pair("AB", chain_roles)
        assert result == "unknown-unknown"

    def test_interface_role_pair_missing_both_chains(self):
        """Test with both chains unmapped."""
        chain_roles = {}
        result = interface_role_pair("AB", chain_roles)
        assert result == "unknown-unknown"

    def test_interface_role_pair_invalid_interface(self):
        """Test with invalid interface string (too short)."""
        chain_roles = {"A": "binder"}
        result = interface_role_pair("A", chain_roles)
        assert result == "unknown-unknown"

    def test_interface_role_pair_empty_interface(self):
        """Test with empty interface string."""
        chain_roles = {"A": "binder"}
        result = interface_role_pair("", chain_roles)
        assert result == "unknown-unknown"


class TestIntegration:
    """Integration tests for sabdab module."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from TSV to interface role pairs."""
        # Create test TSV
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tH\tL\tA\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        # Load role counts
        role_counts = load_sabdab_role_counts(tsv_path)
        assert "1abc" in role_counts

        # Infer chain roles
        chain_ids = ["H", "L", "A"]
        chain_roles = infer_chain_roles(chain_ids, "1abc", role_counts)
        assert chain_roles["H"] == "binder"
        assert chain_roles["L"] == "binder"
        assert chain_roles["A"] == "antigen"

        # Get interface role pairs
        interface_hl = interface_role_pair("HL", chain_roles)
        interface_ha = interface_role_pair("HA", chain_roles)
        interface_la = interface_role_pair("LA", chain_roles)

        assert interface_hl == "binder-binder"
        assert interface_ha == "binder-antigen"
        assert interface_la == "binder-antigen"

    def test_workflow_with_chain_types(self, tmp_path):
        """Test loading and using detailed chain type information."""
        tsv_content = "pdb\tHchain\tLchain\tantigen_chain\n1abc\tH|K\tL\tA|B|C\n"
        tsv_path = tmp_path / "summary.tsv"
        tsv_path.write_text(tsv_content)

        chain_types = load_sabdab_chain_types(tsv_path)
        assert chain_types["1abc"]["H"] == "heavy"
        assert chain_types["1abc"]["K"] == "heavy"
        assert chain_types["1abc"]["L"] == "light"
        assert chain_types["1abc"]["A"] == "antigen"
        assert chain_types["1abc"]["B"] == "antigen"
        assert chain_types["1abc"]["C"] == "antigen"
