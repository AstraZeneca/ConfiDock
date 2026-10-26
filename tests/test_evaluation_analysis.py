"""Tests for confidock.evaluation.analysis module."""

import pandas as pd
import pytest

from confidock.evaluation.analysis import (
    BIAS_REQUIRED_COLS,
    build_condition_label,
    check_bias_columns,
    compute_bias_deltas,
    csv_has_bias_metrics,
    get_grouping_axes,
    infer_chain_topology,
    infer_condition_cols,
)


class TestInferConditionCols:
    """Tests for infer_condition_cols function."""

    def test_infer_condition_cols_boltz_grid(self):
        """Test inference of Boltz grid columns."""
        rows = [
            {"msa_mode": "paired", "recycling_steps": "3", "step_scale": "2.0"},
            {"msa_mode": "unpaired", "recycling_steps": "5", "step_scale": "1.0"},
        ]
        result = infer_condition_cols(rows)
        assert result == ["msa_mode", "recycling_steps", "step_scale"]

    def test_infer_condition_cols_protenix_grid(self):
        """Test inference of Protenix grid columns."""
        rows = [
            {"msa_mode": "paired", "cycle": "4", "step": "48"},
            {"msa_mode": "unpaired", "cycle": "8", "step": "24"},
        ]
        result = infer_condition_cols(rows)
        assert result == ["msa_mode", "cycle", "step"]

    def test_infer_condition_cols_msa_only(self):
        """Test fallback to msa_mode only."""
        rows = [
            {"msa_mode": "paired", "other_col": "value"},
        ]
        result = infer_condition_cols(rows)
        assert result == ["msa_mode"]

    def test_infer_condition_cols_empty_rows(self):
        """Test with empty rows list."""
        result = infer_condition_cols([])
        assert result == []

    def test_infer_condition_cols_no_matching_columns(self):
        """Test with rows that don't match any known grid."""
        rows = [{"foo": "bar", "baz": "qux"}]
        result = infer_condition_cols(rows)
        assert result == []

    def test_infer_condition_cols_partial_boltz_match(self):
        """Test with partial Boltz columns (missing one)."""
        rows = [
            {"msa_mode": "paired", "recycling_steps": "3"},  # Missing step_scale
        ]
        result = infer_condition_cols(rows)
        # Should not match full Boltz grid, fall back to msa_mode
        assert result == ["msa_mode"]


class TestGetGroupingAxes:
    """Tests for get_grouping_axes function."""

    def test_get_grouping_axes_boltz_style(self):
        """Test Boltz-style grouping axes."""
        df = pd.DataFrame({"recycling_steps": [3, 5], "step_scale": [1.0, 2.0]})
        result = get_grouping_axes(df)
        assert result == ["recycling_steps", "step_scale"]

    def test_get_grouping_axes_protenix_style(self):
        """Test Protenix-style grouping axes."""
        df = pd.DataFrame({"cycle": [4, 8], "step": [24, 48]})
        result = get_grouping_axes(df)
        assert result == ["cycle", "step"]

    def test_get_grouping_axes_no_grid_columns(self):
        """Test with no recognized grid columns."""
        df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        result = get_grouping_axes(df)
        assert result == []

    def test_get_grouping_axes_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = get_grouping_axes(df)
        assert result == []


class TestInferChainTopology:
    """Tests for infer_chain_topology function."""

    def test_infer_chain_topology_paired_hclc(self):
        """Test paired_HcLc topology."""
        df = pd.DataFrame({"model_chains": ["Hc_Lc_Ag", "Ag_Hc_Lc"]})
        result = infer_chain_topology(df)
        assert all(result == "paired_HcLc")

    def test_infer_chain_topology_paired_hl(self):
        """Test paired_HL topology."""
        df = pd.DataFrame({"model_chains": ["HL_Ag", "Ag_HL"]})
        result = infer_chain_topology(df)
        assert all(result == "paired_HL")

    def test_infer_chain_topology_single_h_or_l(self):
        """Test single H or L chain."""
        df = pd.DataFrame({"model_chains": ["Hc_Ag", "Lc_Ag", "Ag_Hc"]})
        result = infer_chain_topology(df)
        assert all(result == "single_H_or_L")

    def test_infer_chain_topology_contains_x(self):
        """Test contains x or X chain."""
        df = pd.DataFrame({"model_chains": ["x_Ag", "X_chain", "somex"]})
        result = infer_chain_topology(df)
        assert all(result == "contains_x_chain")

    def test_infer_chain_topology_other(self):
        """Test other chain style."""
        df = pd.DataFrame({"model_chains": ["A_B_C", "Chain1_Chain2"]})
        result = infer_chain_topology(df)
        assert all(result == "other_chain_style")

    def test_infer_chain_topology_missing_column(self):
        """Test with missing model_chains column."""
        df = pd.DataFrame({"other_col": ["value1", "value2"]})
        result = infer_chain_topology(df)
        assert all(result == "other_chain_style")

    def test_infer_chain_topology_null_values(self):
        """Test with null values."""
        df = pd.DataFrame({"model_chains": [None, "", "Hc_Lc"]})
        result = infer_chain_topology(df)
        assert result.iloc[0] == "other_chain_style"
        assert result.iloc[1] == "other_chain_style"
        assert result.iloc[2] == "paired_HcLc"

    def test_infer_chain_topology_priority(self):
        """Test that classification priority is correct (paired_HcLc > paired_HL > single > x > other)."""
        df = pd.DataFrame(
            {
                "model_chains": [
                    "Hc_Lc_x",  # Has Hc+Lc and x, should be paired_HcLc (highest priority)
                    "HL_x",  # Has HL and x, should be paired_HL
                    "Hc_x",  # Has only Hc and x, should be single_H_or_L
                    "x_only",  # Has only x, should be contains_x_chain
                ]
            }
        )
        result = infer_chain_topology(df)
        assert result.iloc[0] == "paired_HcLc"
        assert result.iloc[1] == "paired_HL"
        assert result.iloc[2] == "single_H_or_L"
        assert result.iloc[3] == "contains_x_chain"


class TestCheckBiasColumns:
    """Tests for check_bias_columns function."""

    def test_check_bias_columns_all_present(self):
        """Test with all required columns present."""
        df = pd.DataFrame(columns=list(BIAS_REQUIRED_COLS))
        has_all, missing = check_bias_columns(df)
        assert has_all is True
        assert missing == []

    def test_check_bias_columns_missing_some(self):
        """Test with some columns missing."""
        cols = list(BIAS_REQUIRED_COLS)
        cols.remove("DockQ")
        cols.remove("binder_ca_rmsd_mean")
        df = pd.DataFrame(columns=cols)
        has_all, missing = check_bias_columns(df)
        assert has_all is False
        assert "DockQ" in missing
        assert "binder_ca_rmsd_mean" in missing

    def test_check_bias_columns_none_present(self):
        """Test with no bias columns present."""
        df = pd.DataFrame(columns=["foo", "bar"])
        has_all, missing = check_bias_columns(df)
        assert has_all is False
        assert len(missing) == len(BIAS_REQUIRED_COLS)

    def test_check_bias_columns_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        has_all, missing = check_bias_columns(df)
        assert has_all is False
        assert len(missing) == len(BIAS_REQUIRED_COLS)


class TestCsvHasBiasMetrics:
    """Tests for csv_has_bias_metrics function."""

    def test_csv_has_bias_metrics_valid(self, tmp_path):
        """Test CSV with valid bias metrics."""
        csv_data = {
            "DockQ": [0.5, 0.6, 0.7],
            "binder_ca_rmsd_mean": [1.0, 2.0, 3.0],
            "antigen_ca_rmsd_mean": [1.5, 2.5, 3.5],
            "binder_interface_sidechain_rmsd_mean": [2.0, 3.0, 4.0],
            "antigen_interface_sidechain_rmsd_mean": [2.5, 3.5, 4.5],
            "binder_residue_recall_mean": [0.8, 0.85, 0.9],
            "antigen_residue_recall_mean": [0.7, 0.75, 0.8],
            "interface_contact_f1_mean": [0.6, 0.65, 0.7],
        }
        df = pd.DataFrame(csv_data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        assert csv_has_bias_metrics(csv_path) is True

    def test_csv_has_bias_metrics_missing_columns(self, tmp_path):
        """Test CSV with missing required columns."""
        csv_data = {
            "DockQ": [0.5, 0.6],
            "binder_ca_rmsd_mean": [1.0, 2.0],
        }
        df = pd.DataFrame(csv_data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        assert csv_has_bias_metrics(csv_path) is False

    def test_csv_has_bias_metrics_all_null(self, tmp_path):
        """Test CSV with null values in structural metrics."""
        csv_data = {
            "DockQ": [0.5, 0.6],
            "binder_ca_rmsd_mean": [None, None],
            "antigen_ca_rmsd_mean": [None, None],
            "binder_interface_sidechain_rmsd_mean": [None, None],
            "antigen_interface_sidechain_rmsd_mean": [None, None],
            "binder_residue_recall_mean": [None, None],
            "antigen_residue_recall_mean": [None, None],
            "interface_contact_f1_mean": [None, None],
        }
        df = pd.DataFrame(csv_data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        assert csv_has_bias_metrics(csv_path) is False

    def test_csv_has_bias_metrics_insufficient_non_null(self, tmp_path):
        """Test CSV with only 2 non-null structural columns (need 3+)."""
        csv_data = {
            "DockQ": [0.5, 0.6],
            "binder_ca_rmsd_mean": [1.0, 2.0],
            "antigen_ca_rmsd_mean": [1.5, 2.5],
            "binder_interface_sidechain_rmsd_mean": [None, None],
            "antigen_interface_sidechain_rmsd_mean": [None, None],
            "binder_residue_recall_mean": [None, None],
            "antigen_residue_recall_mean": [None, None],
            "interface_contact_f1_mean": [None, None],
        }
        df = pd.DataFrame(csv_data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        assert csv_has_bias_metrics(csv_path) is False

    def test_csv_has_bias_metrics_nonexistent_file(self, tmp_path):
        """Test with non-existent file."""
        assert csv_has_bias_metrics(tmp_path / "nonexistent.csv") is False


class TestComputeBiasDeltas:
    """Tests for compute_bias_deltas function."""

    def test_compute_bias_deltas_basic(self):
        """Test basic delta computation."""
        df = pd.DataFrame(
            {
                "DockQ": [0.5, 0.6],
                "binder_ca_rmsd_mean": [1.0, 2.0],
                "antigen_ca_rmsd_mean": [2.0, 3.0],
                "binder_interface_sidechain_rmsd_mean": [1.5, 2.5],
                "antigen_interface_sidechain_rmsd_mean": [2.5, 3.5],
                "binder_residue_recall_mean": [0.8, 0.9],
                "antigen_residue_recall_mean": [0.7, 0.75],
                "interface_contact_f1_mean": [0.6, 0.65],
            }
        )

        result = compute_bias_deltas(df)

        # Check delta columns exist
        assert "delta_ca_rmsd_antigen_minus_binder" in result.columns
        assert "delta_sc_rmsd_antigen_minus_binder" in result.columns
        assert "delta_recall_binder_minus_antigen" in result.columns

        # Check calculations
        assert result["delta_ca_rmsd_antigen_minus_binder"].iloc[0] == pytest.approx(1.0)  # 2.0 - 1.0
        assert result["delta_ca_rmsd_antigen_minus_binder"].iloc[1] == pytest.approx(1.0)  # 3.0 - 2.0
        assert result["delta_sc_rmsd_antigen_minus_binder"].iloc[0] == pytest.approx(1.0)  # 2.5 - 1.5
        assert result["delta_recall_binder_minus_antigen"].iloc[0] == pytest.approx(0.1)  # 0.8 - 0.7

    def test_compute_bias_deltas_with_nulls(self):
        """Test delta computation with null values."""
        df = pd.DataFrame(
            {
                "DockQ": [0.5, 0.6],
                "binder_ca_rmsd_mean": [1.0, None],
                "antigen_ca_rmsd_mean": [2.0, 3.0],
                "binder_interface_sidechain_rmsd_mean": [1.5, 2.5],
                "antigen_interface_sidechain_rmsd_mean": [2.5, None],
                "binder_residue_recall_mean": [0.8, 0.9],
                "antigen_residue_recall_mean": [0.7, 0.75],
                "interface_contact_f1_mean": [0.6, 0.65],
            }
        )

        result = compute_bias_deltas(df)

        # First row should be valid
        assert result["delta_ca_rmsd_antigen_minus_binder"].iloc[0] == pytest.approx(1.0)
        # Second row should have NaN for ca_rmsd delta
        assert pd.isna(result["delta_ca_rmsd_antigen_minus_binder"].iloc[1])
        assert pd.isna(result["delta_sc_rmsd_antigen_minus_binder"].iloc[1])


class TestBuildConditionLabel:
    """Tests for build_condition_label function."""

    def test_build_condition_label_boltz_style(self):
        """Test Boltz-style condition label building."""
        df = pd.DataFrame(
            {
                "msa_mode": ["paired", "unpaired"],
                "recycling_steps": [3, 5],
                "step_scale": [1.0, 2.0],
            }
        )

        result = build_condition_label(df)

        assert "condition" in result.columns
        assert result["condition"].iloc[0] == "msa=paired__recycling=3__scale=1.0"
        assert result["condition"].iloc[1] == "msa=unpaired__recycling=5__scale=2.0"

    def test_build_condition_label_protenix_style(self):
        """Test Protenix-style condition label building."""
        df = pd.DataFrame(
            {
                "msa_mode": ["paired", "unpaired"],
                "cycle": [4, 8],
                "step": [24, 48],
            }
        )

        result = build_condition_label(df)

        assert "condition" in result.columns
        assert result["condition"].iloc[0] == "msa=paired__cycle=4__step=24"
        assert result["condition"].iloc[1] == "msa=unpaired__cycle=8__step=48"

    def test_build_condition_label_already_exists(self):
        """Test when condition column already exists."""
        df = pd.DataFrame(
            {
                "condition": ["existing1", "existing2"],
                "msa_mode": ["paired", "unpaired"],
                "recycling_steps": [3, 5],
                "step_scale": [1.0, 2.0],
            }
        )

        result = build_condition_label(df)

        # Should not overwrite existing condition
        assert result["condition"].iloc[0] == "existing1"
        assert result["condition"].iloc[1] == "existing2"

    def test_build_condition_label_no_grid_columns(self):
        """Test with no recognized grid columns."""
        df = pd.DataFrame(
            {
                "foo": [1, 2],
                "bar": [3, 4],
            }
        )

        result = build_condition_label(df)

        # Should return DataFrame unchanged
        assert "condition" not in result.columns

    def test_build_condition_label_partial_columns(self):
        """Test with partial grid columns (e.g., only msa_mode and recycling_steps)."""
        df = pd.DataFrame(
            {
                "msa_mode": ["paired", "unpaired"],
                "recycling_steps": [3, 5],
            }
        )

        result = build_condition_label(df)

        # Should not add condition if not all grid columns present
        assert "condition" not in result.columns


class TestIntegration:
    """Integration tests for analysis module."""

    def test_full_bias_analysis_workflow(self, tmp_path):
        """Test complete bias analysis workflow."""
        # Create CSV with bias metrics
        csv_data = {
            "pdb_id": ["1abc", "2def", "3ghi"],
            "msa_mode": ["paired", "paired", "unpaired"],
            "recycling_steps": [3, 3, 5],
            "step_scale": [1.0, 2.0, 1.0],
            "model_chains": ["Hc_Lc_Ag", "HL_Ag", "Hc_Ag"],
            "DockQ": [0.5, 0.6, 0.7],
            "binder_ca_rmsd_mean": [1.0, 2.0, 3.0],
            "antigen_ca_rmsd_mean": [2.0, 3.0, 4.0],
            "binder_interface_sidechain_rmsd_mean": [1.5, 2.5, 3.5],
            "antigen_interface_sidechain_rmsd_mean": [2.5, 3.5, 4.5],
            "binder_residue_recall_mean": [0.8, 0.85, 0.9],
            "antigen_residue_recall_mean": [0.7, 0.75, 0.8],
            "interface_contact_f1_mean": [0.6, 0.65, 0.7],
        }
        df = pd.DataFrame(csv_data)
        csv_path = tmp_path / "bias_data.csv"
        df.to_csv(csv_path, index=False)

        # Check if CSV has bias metrics
        assert csv_has_bias_metrics(csv_path) is True

        # Load and process
        df = pd.read_csv(csv_path)

        # Check required columns
        has_all, _ = check_bias_columns(df)
        assert has_all is True

        # Compute deltas
        df = compute_bias_deltas(df)
        assert "delta_ca_rmsd_antigen_minus_binder" in df.columns

        # Build condition labels
        df = build_condition_label(df)
        assert "condition" in df.columns

        # Infer chain topology
        df["chain_topology"] = infer_chain_topology(df)
        assert df["chain_topology"].iloc[0] == "paired_HcLc"
        assert df["chain_topology"].iloc[1] == "paired_HL"
        assert df["chain_topology"].iloc[2] == "single_H_or_L"

        # Get grouping axes
        axes = get_grouping_axes(df)
        assert axes == ["recycling_steps", "step_scale"]
