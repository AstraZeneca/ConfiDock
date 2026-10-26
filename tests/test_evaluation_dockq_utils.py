"""Tests for confidock.evaluation.dockq_utils module."""

import pytest

from confidock.evaluation.dockq_utils import (
    BOLTZ_CONDITION_COLS,
    ESMFOLD2_CONDITION_COLS,
    GRID_SCHEMAS,
    PROTENIX_CONDITION_COLS,
    _build_candidates_from_condition_dir,
    _build_candidates_from_grid_dir,
    _build_candidates_from_predictions_dir,
    _contains_entry_yaml_files,
    _is_schema_consistent,
    _match_directory_to_schema,
    _should_skip_condition,
    format_value,
    make_dockq_row,
    resolve_boltz_input_dir,
    resolve_grid_predictions_dir,
    scan_grid_conditions,
    validate_interface_coverage,
)


class TestFormatValue:
    """Tests for format_value function."""

    def test_format_value_float(self):
        """Test formatting float values."""
        assert format_value(1.23456789) == pytest.approx(1.234568)
        assert format_value(0.123456) == pytest.approx(0.123456)

    def test_format_value_none(self):
        """Test formatting None values."""
        assert format_value(None) == ""

    def test_format_value_string(self):
        """Test formatting string values."""
        assert format_value("test") == "test"

    def test_format_value_int(self):
        """Test formatting integer values."""
        assert format_value(42) == 42

    def test_format_value_bool(self):
        """Test formatting boolean values."""
        assert format_value(True) is True
        assert format_value(False) is False


class TestResolveGridPredictionsDir:
    """Tests for resolve_grid_predictions_dir function."""

    def test_resolve_grid_predictions_dir_boltz_standard(self, tmp_path):
        """Test resolving Boltz standard layout."""
        condition_dir = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        preds_dir = condition_dir / "boltz_results__staged_inputs" / "predictions" / "entry1"
        preds_dir.mkdir(parents=True)

        result = resolve_grid_predictions_dir(condition_dir, "boltz")
        assert result == condition_dir / "boltz_results__staged_inputs" / "predictions"

    def test_resolve_grid_predictions_dir_boltz_simple(self, tmp_path):
        """Test resolving Boltz simple layout."""
        condition_dir = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        preds_dir = condition_dir / "predictions" / "entry1"
        preds_dir.mkdir(parents=True)

        result = resolve_grid_predictions_dir(condition_dir, "boltz")
        assert result == condition_dir / "predictions"

    def test_resolve_grid_predictions_dir_boltz_direct(self, tmp_path):
        """Test resolving Boltz direct layout (entries in condition_dir)."""
        condition_dir = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        entry_dir = condition_dir / "entry1"
        entry_dir.mkdir(parents=True)

        result = resolve_grid_predictions_dir(condition_dir, "boltz")
        assert result == condition_dir

    def test_resolve_grid_predictions_dir_protenix(self, tmp_path):
        """Test resolving Protenix layout."""
        condition_dir = tmp_path / "msa=paired__cycle=4__step=48"
        preds_dir = condition_dir / "predictions" / "entry1"
        preds_dir.mkdir(parents=True)

        result = resolve_grid_predictions_dir(condition_dir, "protenix")
        assert result == condition_dir / "predictions"

    def test_resolve_grid_predictions_dir_not_found(self, tmp_path):
        """Test when no predictions directory found."""
        condition_dir = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        condition_dir.mkdir()

        result = resolve_grid_predictions_dir(condition_dir, "boltz")
        assert result is None


class TestContainsEntryYamlFiles:
    """Tests for _contains_entry_yaml_files function."""

    def test_contains_entry_yaml_files_yaml_extension(self, tmp_path):
        """Test finding entry YAML files with .yaml extension."""
        (tmp_path / "8qf5_H_A.yaml").touch()
        assert _contains_entry_yaml_files(tmp_path) is True

    def test_contains_entry_yaml_files_yml_extension(self, tmp_path):
        """Test finding entry YAML files with .yml extension."""
        (tmp_path / "1abc_H_L_A.yml").touch()
        assert _contains_entry_yaml_files(tmp_path) is True

    def test_contains_entry_yaml_files_config_only(self, tmp_path):
        """Test that config files are not considered entry files."""
        (tmp_path / "grid_config.yaml").touch()
        (tmp_path / "hparams.yaml").touch()
        assert _contains_entry_yaml_files(tmp_path) is False

    def test_contains_entry_yaml_files_nonexistent(self, tmp_path):
        """Test with non-existent directory."""
        assert _contains_entry_yaml_files(tmp_path / "nonexistent") is False

    def test_contains_entry_yaml_files_empty_dir(self, tmp_path):
        """Test with empty directory."""
        assert _contains_entry_yaml_files(tmp_path) is False


class TestBuildCandidates:
    """Tests for candidate directory building functions."""

    def test_build_candidates_from_grid_dir_boltz2(self, tmp_path):
        """Test building candidates from Boltz2 grid directory."""
        grid_dir = tmp_path / "dataset_boltz2_grid"
        candidates = _build_candidates_from_grid_dir(grid_dir)

        assert len(candidates) == 1
        assert candidates[0] == tmp_path / "dataset_boltz2_inputs"

    def test_build_candidates_from_grid_dir_generic(self, tmp_path):
        """Test building candidates from generic grid directory."""
        grid_dir = tmp_path / "dataset_grid"
        candidates = _build_candidates_from_grid_dir(grid_dir)

        assert len(candidates) == 1
        assert candidates[0] == tmp_path / "dataset_inputs"

    def test_build_candidates_from_grid_dir_none(self):
        """Test with None grid_dir."""
        candidates = _build_candidates_from_grid_dir(None)
        assert candidates == []

    def test_build_candidates_from_condition_dir(self, tmp_path):
        """Test building candidates from condition directory."""
        condition_dir = tmp_path / "msa=full__recycling=3__scale=1.0"
        candidates = _build_candidates_from_condition_dir(condition_dir)

        assert len(candidates) == 2
        assert condition_dir / "_staged_inputs" in candidates
        assert condition_dir / "staged_inputs" in candidates

    def test_build_candidates_from_condition_dir_none(self):
        """Test with None condition_dir."""
        candidates = _build_candidates_from_condition_dir(None)
        assert candidates == []

    def test_build_candidates_from_predictions_dir(self, tmp_path):
        """Test building candidates from predictions directory."""
        predictions_dir = tmp_path / "parent" / "predictions"
        candidates = _build_candidates_from_predictions_dir(predictions_dir)

        assert len(candidates) == 3
        assert predictions_dir.parent / "_staged_inputs" in candidates
        assert predictions_dir.parent.parent / "_staged_inputs" in candidates
        assert predictions_dir.parent.parent / "staged_inputs" in candidates

    def test_build_candidates_from_predictions_dir_wrong_name(self, tmp_path):
        """Test with directory not named 'predictions'."""
        other_dir = tmp_path / "other"
        candidates = _build_candidates_from_predictions_dir(other_dir)
        assert candidates == []


class TestResolveBoltzInputDir:
    """Tests for resolve_boltz_input_dir function."""

    def test_resolve_boltz_input_dir_explicit(self, tmp_path):
        """Test with explicit input directory."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "8qf5_H_A.yaml").touch()

        result = resolve_boltz_input_dir(predictions_dir=tmp_path / "predictions", explicit_input_dir=input_dir)

        assert result == input_dir

    def test_resolve_boltz_input_dir_from_grid_pattern(self, tmp_path):
        """Test inference from grid directory pattern."""
        grid_dir = tmp_path / "dataset_boltz2_grid"
        input_dir = tmp_path / "dataset_boltz2_inputs"
        input_dir.mkdir()
        (input_dir / "8qf5_H_A.yaml").touch()

        result = resolve_boltz_input_dir(predictions_dir=tmp_path / "predictions", grid_dir=grid_dir)

        assert result == input_dir

    def test_resolve_boltz_input_dir_from_condition(self, tmp_path):
        """Test inference from condition directory."""
        condition_dir = tmp_path / "msa=full__recycling=3__scale=1.0"
        input_dir = condition_dir / "_staged_inputs"
        input_dir.mkdir(parents=True)
        (input_dir / "8qf5_H_A.yaml").touch()

        result = resolve_boltz_input_dir(predictions_dir=tmp_path / "predictions", condition_dir=condition_dir)

        assert result == input_dir

    def test_resolve_boltz_input_dir_not_found(self, tmp_path):
        """Test when no input directory found."""
        result = resolve_boltz_input_dir(predictions_dir=tmp_path / "predictions")

        assert result is None

    def test_resolve_boltz_input_dir_explicit_invalid(self, tmp_path):
        """Test that invalid explicit_input_dir is ignored and fallback to pattern."""
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        # No YAML files

        # Create the expected pattern-based directory
        grid_dir = tmp_path / "dataset_grid"
        grid_dir.mkdir()
        inputs_dir = tmp_path / "dataset_inputs"
        inputs_dir.mkdir()
        (inputs_dir / "8qf5_H_A.yaml").touch()

        result = resolve_boltz_input_dir(
            predictions_dir=tmp_path / "predictions", explicit_input_dir=invalid_dir, grid_dir=grid_dir
        )

        # Should skip invalid explicit and find from pattern
        assert result == inputs_dir


class TestMatchDirectoryToSchema:
    """Tests for _match_directory_to_schema function."""

    def test_match_directory_boltz_schema(self):
        """Test matching Boltz schema."""
        schema, meta = _match_directory_to_schema("msa=full_msa__recycling=3__scale=1.0")

        assert schema is not None
        assert schema["name"] == "boltz"
        assert meta["msa_mode"] == "full_msa"
        assert meta["recycling_steps"] == 3
        assert meta["step_scale"] == pytest.approx(1.0)

    def test_match_directory_protenix_schema(self):
        """Test matching Protenix schema."""
        schema, meta = _match_directory_to_schema("msa=paired__cycle=4__step=48")

        assert schema is not None
        assert schema["name"] == "protenix"
        assert meta["msa_mode"] == "paired"
        assert meta["cycle"] == 4
        assert meta["step"] == 48

    def test_match_directory_esmfold2_schema(self):
        """Test matching ESMFold2 schema."""
        schema, meta = _match_directory_to_schema("msa=full__loops=2__steps=50")

        assert schema is not None
        assert schema["name"] == "esmfold2"
        assert meta["msa_mode"] == "full"
        assert meta["loops"] == 2
        assert meta["steps"] == 50

    def test_match_directory_no_match(self):
        """Test with non-matching directory name."""
        schema, meta = _match_directory_to_schema("random_directory")

        assert schema is None
        assert meta is None


class TestShouldSkipCondition:
    """Tests for _should_skip_condition function."""

    def test_should_skip_condition_filtered(self):
        """Test skipping condition based on MSA mode filter."""
        meta = {"msa_mode": "paired"}
        assert _should_skip_condition(meta, ["full_msa", "unpaired"]) is True

    def test_should_skip_condition_allowed(self):
        """Test not skipping when MSA mode is in filter."""
        meta = {"msa_mode": "paired"}
        assert _should_skip_condition(meta, ["paired", "full_msa"]) is False

    def test_should_skip_condition_no_filter(self):
        """Test not skipping when no filter provided."""
        meta = {"msa_mode": "paired"}
        assert _should_skip_condition(meta, None) is False


class TestIsSchemaConsistent:
    """Tests for _is_schema_consistent function."""

    def test_is_schema_consistent_matching(self):
        """Test with matching schemas."""
        matched_cols = ["msa_mode", "recycling_steps", "step_scale"]
        condition_cols = ["msa_mode", "recycling_steps", "step_scale"]
        schema = {"name": "boltz"}

        assert _is_schema_consistent(matched_cols, condition_cols, "boltz", schema, "test_dir") is True

    def test_is_schema_consistent_mismatched(self):
        """Test with mismatched schemas."""
        matched_cols = ["msa_mode", "cycle", "step"]
        condition_cols = ["msa_mode", "recycling_steps", "step_scale"]
        schema = {"name": "protenix"}

        assert _is_schema_consistent(matched_cols, condition_cols, "boltz", schema, "test_dir") is False

    def test_is_schema_consistent_first_match(self):
        """Test with no previous schema (first match)."""
        matched_cols = ["msa_mode", "recycling_steps", "step_scale"]
        condition_cols = []
        schema = {"name": "boltz"}

        assert _is_schema_consistent(matched_cols, condition_cols, None, schema, "test_dir") is True


class TestScanGridConditions:
    """Tests for scan_grid_conditions function."""

    def test_scan_grid_conditions_boltz(self, tmp_path):
        """Test scanning Boltz grid conditions."""
        # Create grid structure
        cond1 = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        cond1_preds = cond1 / "predictions" / "entry1"
        cond1_preds.mkdir(parents=True)

        cond2 = tmp_path / "msa=paired__recycling=5__scale=2.0"
        cond2_preds = cond2 / "predictions" / "entry1"
        cond2_preds.mkdir(parents=True)

        results, cols, schema_name = scan_grid_conditions(tmp_path, "boltz")

        assert len(results) == 2
        assert cols == BOLTZ_CONDITION_COLS
        assert schema_name == "boltz"

        # Check metadata
        metas = [r[0] for r in results]
        assert any(m["msa_mode"] == "full_msa" for m in metas)
        assert any(m["msa_mode"] == "paired" for m in metas)

    def test_scan_grid_conditions_with_filter(self, tmp_path):
        """Test scanning with MSA mode filter."""
        cond1 = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        cond1_preds = cond1 / "predictions" / "entry1"
        cond1_preds.mkdir(parents=True)

        cond2 = tmp_path / "msa=paired__recycling=5__scale=2.0"
        cond2_preds = cond2 / "predictions" / "entry1"
        cond2_preds.mkdir(parents=True)

        results, _, _ = scan_grid_conditions(tmp_path, "boltz", msa_modes=["full_msa"])

        assert len(results) == 1
        assert results[0][0]["msa_mode"] == "full_msa"

    def test_scan_grid_conditions_empty_grid(self, tmp_path):
        """Test scanning empty grid directory."""
        results, cols, schema_name = scan_grid_conditions(tmp_path, "boltz")

        assert results == []
        assert cols == []
        assert schema_name is None

    def test_scan_grid_conditions_no_predictions(self, tmp_path):
        """Test scanning with conditions but no predictions directories."""
        cond1 = tmp_path / "msa=full_msa__recycling=3__scale=1.0"
        cond1.mkdir()
        # No predictions subdirectory

        results, _, _ = scan_grid_conditions(tmp_path, "boltz")

        assert results == []


class TestMakeDockqRow:
    """Tests for make_dockq_row function."""

    def test_make_dockq_row_basic(self, tmp_path):
        """Test creating basic DockQ row."""
        model_path = tmp_path / "model.pdb"
        native_path = tmp_path / "native.pdb"

        metrics = {
            "DockQ": 0.75,
            "iRMSD": 1.5,
            "LRMSD": 2.0,
            "fnat": 0.85,
            "fnonnat": 0.1,
            "F1": 0.87,
            "clashes": 2,
            "model_chains": "AB",
        }

        confidence = {"plddt": 85.5, "ptm": 0.80}
        confidence_cols = ["plddt", "ptm"]

        row = make_dockq_row(
            entry="8qf5",
            seed=1,
            sample_idx=0,
            model_path=model_path,
            native_path=native_path,
            iface_key="AB",
            metrics=metrics,
            global_dockq=0.75,
            confidence=confidence,
            confidence_cols=confidence_cols,
        )

        assert row["entry"] == "8qf5"
        assert row["seed"] == 1
        assert row["sample"] == 0
        assert row["native_interface"] == "AB"
        assert row["DockQ"] == pytest.approx(0.75)
        assert row["plddt"] == pytest.approx(85.5)

    def test_make_dockq_row_no_seed(self, tmp_path):
        """Test with None seed."""
        model_path = tmp_path / "model.pdb"
        native_path = tmp_path / "native.pdb"

        metrics = {"DockQ": 0.75, "model_chains": "AB"}
        confidence = {}
        confidence_cols = []

        row = make_dockq_row(
            entry="8qf5",
            seed=None,
            sample_idx=0,
            model_path=model_path,
            native_path=native_path,
            iface_key="AB",
            metrics=metrics,
            global_dockq=0.75,
            confidence=confidence,
            confidence_cols=confidence_cols,
        )

        assert row["seed"] == ""

    def test_make_dockq_row_with_structure_metrics(self, tmp_path):
        """Test with additional structure metrics."""
        model_path = tmp_path / "model.pdb"
        native_path = tmp_path / "native.pdb"

        metrics = {"DockQ": 0.75, "model_chains": "AB"}
        confidence = {}
        structure_metrics = {"binder_ca_rmsd": 1.2, "antigen_ca_rmsd": 1.5}

        row = make_dockq_row(
            entry="8qf5",
            seed=1,
            sample_idx=0,
            model_path=model_path,
            native_path=native_path,
            iface_key="AB",
            metrics=metrics,
            global_dockq=0.75,
            confidence=confidence,
            confidence_cols=[],
            structure_metrics=structure_metrics,
        )

        assert row["binder_ca_rmsd"] == pytest.approx(1.2)
        assert row["antigen_ca_rmsd"] == pytest.approx(1.5)

    def test_make_dockq_row_with_complex_metrics(self, tmp_path):
        """Test with complex-level metrics."""
        model_path = tmp_path / "model.pdb"
        native_path = tmp_path / "native.pdb"

        metrics = {"DockQ": 0.75, "model_chains": "AB"}
        confidence = {}
        complex_metrics = {"complex_ca_rmsd": 2.0}
        complex_cols = ["complex_ca_rmsd"]

        row = make_dockq_row(
            entry="8qf5",
            seed=1,
            sample_idx=0,
            model_path=model_path,
            native_path=native_path,
            iface_key="AB",
            metrics=metrics,
            global_dockq=0.75,
            confidence=confidence,
            confidence_cols=[],
            complex_metrics=complex_metrics,
            complex_cols=complex_cols,
        )

        assert row["complex_ca_rmsd"] == pytest.approx(2.0)


class TestValidateInterfaceCoverage:
    """Tests for validate_interface_coverage function."""

    def test_validate_interface_coverage_complete(self):
        """Test with complete interface coverage."""
        interfaces = {"AB": {}, "AC": {}, "BC": {}}
        native_chain_roles = {"A": "binder", "B": "binder", "C": "antigen"}

        missing = validate_interface_coverage(interfaces, native_chain_roles, "8qf5")

        assert missing == []

    def test_validate_interface_coverage_missing(self):
        """Test with missing interfaces."""
        interfaces = {"AB": {}}  # Missing AC and BC
        native_chain_roles = {"A": "binder", "B": "binder", "C": "antigen"}

        missing = validate_interface_coverage(interfaces, native_chain_roles, "8qf5")

        assert len(missing) == 2
        assert "AC" in missing
        assert "BC" in missing

    def test_validate_interface_coverage_two_chains(self):
        """Test with two-chain complex."""
        interfaces = {"AB": {}}
        native_chain_roles = {"A": "binder", "B": "antigen"}

        missing = validate_interface_coverage(interfaces, native_chain_roles, "8qf5")

        assert missing == []

    def test_validate_interface_coverage_single_chain(self):
        """Test with single chain (no interfaces)."""
        interfaces = {}
        native_chain_roles = {"A": "binder"}

        missing = validate_interface_coverage(interfaces, native_chain_roles, "8qf5")

        assert missing == []


class TestGridSchemas:
    """Tests for GRID_SCHEMAS constant."""

    def test_grid_schemas_completeness(self):
        """Test that all expected schemas are defined."""
        schema_names = [s["name"] for s in GRID_SCHEMAS]

        assert "boltz" in schema_names
        assert "protenix" in schema_names
        assert "esmfold2" in schema_names

    def test_grid_schemas_patterns(self):
        """Test that schema patterns work."""
        for schema in GRID_SCHEMAS:
            assert "pattern" in schema
            assert "condition_cols" in schema
            assert "meta_parser" in schema
            assert callable(schema["meta_parser"])


class TestConditionCols:
    """Tests for condition column constants."""

    def test_boltz_condition_cols(self):
        """Test Boltz condition columns."""
        assert BOLTZ_CONDITION_COLS == ["msa_mode", "recycling_steps", "step_scale"]

    def test_protenix_condition_cols(self):
        """Test Protenix condition columns."""
        assert PROTENIX_CONDITION_COLS == ["msa_mode", "cycle", "step"]

    def test_esmfold2_condition_cols(self):
        """Test ESMFold2 condition columns."""
        assert ESMFOLD2_CONDITION_COLS == ["msa_mode", "loops", "steps"]
