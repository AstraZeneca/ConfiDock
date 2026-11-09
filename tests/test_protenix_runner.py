"""Tests for confidock.protenix.runner module."""

from __future__ import annotations

from confidock.protenix.inference.runner import (
    PROTENIX_VERSION_PYPI,
    PROTENIX_VERSION_SHARED,
    check_completed_samples,
    check_completion,
)


class TestCheckCompletion:
    """Tests for check_completion function."""

    def test_no_entries_complete(self, tmp_path):
        """Test when no entries are complete."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = check_completion(output_dir, ["entry1", "entry2"], [0, 1], 2)

        assert result == 0

    def test_all_entries_complete(self, tmp_path):
        """Test when all entries are complete."""
        output_dir = tmp_path / "output"

        # Create complete output structure
        for name in ["entry1", "entry2"]:
            for seed in [0, 1]:
                pred_dir = output_dir / name / f"seed_{seed}" / "predictions"
                pred_dir.mkdir(parents=True)
                for rank in range(2):
                    (pred_dir / f"{name}_sample_{rank}.cif").touch()

        result = check_completion(output_dir, ["entry1", "entry2"], [0, 1], 2)

        assert result == 2

    def test_partial_completion(self, tmp_path):
        """Test when some entries are complete."""
        output_dir = tmp_path / "output"

        # Complete entry1
        for seed in [0, 1]:
            pred_dir = output_dir / "entry1" / f"seed_{seed}" / "predictions"
            pred_dir.mkdir(parents=True)
            for rank in range(2):
                (pred_dir / f"entry1_sample_{rank}.cif").touch()

        # Incomplete entry2 (missing one file)
        for seed in [0, 1]:
            pred_dir = output_dir / "entry2" / f"seed_{seed}" / "predictions"
            pred_dir.mkdir(parents=True)
            (pred_dir / "entry2_sample_0.cif").touch()
            # Missing sample_1

        result = check_completion(output_dir, ["entry1", "entry2"], [0, 1], 2)

        assert result == 1

    def test_single_sample(self, tmp_path):
        """Test with single sample per seed."""
        output_dir = tmp_path / "output"

        pred_dir = output_dir / "entry1" / "seed_0" / "predictions"
        pred_dir.mkdir(parents=True)
        (pred_dir / "entry1_sample_0.cif").touch()

        result = check_completion(output_dir, ["entry1"], [0], 1)

        assert result == 1


class TestCheckCompletedSamples:
    """Tests for check_completed_samples function."""

    def test_no_samples_complete(self, tmp_path):
        """Test when no samples are complete."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = check_completed_samples(output_dir, [0, 1], 2)

        assert result == set()

    def test_all_samples_complete(self, tmp_path):
        """Test when all samples are complete."""
        output_dir = tmp_path / "output"

        # Create complete output structure
        for name in ["sample1", "sample2"]:
            sample_dir = output_dir / name
            for seed in [0, 1]:
                pred_dir = sample_dir / f"seed_{seed}" / "predictions"
                pred_dir.mkdir(parents=True)
                for rank in range(3):
                    (pred_dir / f"{name}_sample_{rank}.cif").touch()

        result = check_completed_samples(output_dir, [0, 1], 3)

        assert result == {"sample1", "sample2"}

    def test_partial_completion(self, tmp_path):
        """Test when some samples are complete."""
        output_dir = tmp_path / "output"

        # Complete sample1
        sample_dir = output_dir / "sample1"
        for seed in [0, 1]:
            pred_dir = sample_dir / f"seed_{seed}" / "predictions"
            pred_dir.mkdir(parents=True)
            for rank in range(2):
                (pred_dir / f"sample1_sample_{rank}.cif").touch()

        # Incomplete sample2 (missing seed 1)
        sample_dir = output_dir / "sample2"
        pred_dir = sample_dir / "seed_0" / "predictions"
        pred_dir.mkdir(parents=True)
        for rank in range(2):
            (pred_dir / f"sample2_sample_{rank}.cif").touch()

        result = check_completed_samples(output_dir, [0, 1], 2)

        assert result == {"sample1"}

    def test_ignores_non_directories(self, tmp_path):
        """Test that non-directory items are ignored."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a file in output_dir (should be ignored)
        (output_dir / "somefile.txt").touch()

        # Create a complete sample
        sample_dir = output_dir / "sample1"
        pred_dir = sample_dir / "seed_0" / "predictions"
        pred_dir.mkdir(parents=True)
        (pred_dir / "sample1_sample_0.cif").touch()

        result = check_completed_samples(output_dir, [0], 1)

        assert result == {"sample1"}

    def test_multiple_seeds_and_samples(self, tmp_path):
        """Test with multiple seeds and samples."""
        output_dir = tmp_path / "output"

        # Create complete output for sample1 with 3 seeds and 5 samples each
        sample_dir = output_dir / "sample1"
        for seed in [100, 200, 300]:
            pred_dir = sample_dir / f"seed_{seed}" / "predictions"
            pred_dir.mkdir(parents=True)
            for rank in range(5):
                (pred_dir / f"sample1_sample_{rank}.cif").touch()

        result = check_completed_samples(output_dir, [100, 200, 300], 5)

        assert result == {"sample1"}


class TestConstants:
    """Tests for module constants."""

    def test_version_constants_exist(self):
        """Test that version constants are defined."""
        assert PROTENIX_VERSION_PYPI == "pypi_v2.0.0"
        assert PROTENIX_VERSION_SHARED == "shared_v0.1.0"

    def test_version_constants_are_strings(self):
        """Test that version constants are strings."""
        assert isinstance(PROTENIX_VERSION_PYPI, str)
        assert isinstance(PROTENIX_VERSION_SHARED, str)


class TestWritePredictionConfig:
    """Tests for write_prediction_config function."""

    def test_write_yaml_config(self, tmp_path):
        """Test writing YAML configuration."""
        import yaml

        from confidock.protenix.inference.runner import write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        write_prediction_config(output_dir, format="yaml", model="protenix", batch_size=8)

        cfg_path = output_dir / "prediction_config.yaml"
        assert cfg_path.exists()

        with cfg_path.open() as fh:
            data = yaml.safe_load(fh)

        assert data["batch_size"] == 8
        assert data["model"] == "protenix"

    def test_write_json_config(self, tmp_path):
        """Test writing JSON configuration."""
        import json

        from confidock.protenix.inference.runner import write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        write_prediction_config(output_dir, format="json", model="protenix", seed=42)

        cfg_path = output_dir / "prediction_config.json"
        assert cfg_path.exists()

        with cfg_path.open() as fh:
            data = json.load(fh)

        assert data["model"] == "protenix"
        assert data["seed"] == 42

    def test_path_conversion(self, tmp_path):
        """Test that Path objects are converted to strings."""
        from pathlib import Path

        import yaml

        from confidock.protenix.inference.runner import write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        input_path = Path("/some/input/path")
        write_prediction_config(output_dir, format="yaml", input_dir=input_path)

        cfg_path = output_dir / "prediction_config.yaml"
        with cfg_path.open() as fh:
            data = yaml.safe_load(fh)

        assert isinstance(data["input_dir"], str)
        assert data["input_dir"] == str(input_path)

    def test_sorted_keys(self, tmp_path):
        """Test that keys are sorted in output."""
        from confidock.protenix.inference.runner import write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        write_prediction_config(output_dir, format="yaml", zebra=1, apple=2, middle=3)

        cfg_path = output_dir / "prediction_config.yaml"
        with cfg_path.open() as fh:
            content = fh.read()

        # Keys should appear in sorted order
        assert content.index("apple") < content.index("middle")
        assert content.index("middle") < content.index("zebra")


class TestPrintTimings:
    """Tests for print_timings function."""

    def test_print_timings(self, caplog):
        """Test printing timing statistics."""
        import logging

        from confidock.protenix.inference.runner import print_timings

        timings = {
            "MSA generation": 100.5,
            "Inference": 200.3,
            "Post-processing": 50.1,
        }

        with caplog.at_level(logging.INFO):
            print_timings(timings)

        # Check that all steps are logged
        assert "MSA generation" in caplog.text
        assert "Inference" in caplog.text
        assert "Post-processing" in caplog.text
        assert "Total" in caplog.text

    def test_print_empty_timings(self, caplog):
        """Test printing empty timings dict."""
        import logging

        from confidock.protenix.inference.runner import print_timings

        with caplog.at_level(logging.INFO):
            print_timings({})

        assert "Timings" in caplog.text

    def test_print_single_timing(self, caplog):
        """Test printing a single timing."""
        import logging

        from confidock.protenix.inference.runner import print_timings

        timings = {"Step 1": 42.7}

        with caplog.at_level(logging.INFO):
            print_timings(timings)

        assert "Step 1" in caplog.text
        assert "42.7" in caplog.text


class TestFindProtenixCli:
    """Tests for find_protenix_cli function."""

    def test_find_protenix_cli_returns_tuple(self):
        """Test that find_protenix_cli returns a tuple of three strings."""
        from confidock.protenix.inference.runner import find_protenix_cli

        cli_command, pythonpath, version = find_protenix_cli()

        assert isinstance(cli_command, str)
        assert isinstance(pythonpath, str)
        assert isinstance(version, str)
        assert len(cli_command) > 0
        assert version in [PROTENIX_VERSION_PYPI, PROTENIX_VERSION_SHARED]
