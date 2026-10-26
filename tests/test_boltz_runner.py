"""Tests for confidock.boltz.inference.runner module."""

from __future__ import annotations

import pytest

from confidock.boltz.inference.runner import (
    BoltzPredictConfig,
    build_command,
    fmt_duration,
    predictions_dir,
    rename_model_outputs,
)


class TestBoltzPredictConfig:
    """Tests for BoltzPredictConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = BoltzPredictConfig()
        assert config.recycling_steps == 3
        assert config.sampling_steps == 200
        assert config.diffusion_samples == 1
        assert config.output_format == "pdb"
        assert config.write_full_pae is False
        assert config.use_potentials is False
        assert config.override is False
        assert config.devices == 1
        assert config.accelerator == "gpu"
        assert config.seed is None
        assert config.use_msa_server is False
        assert config.no_kernels is False
        assert config.step_scale is None
        assert config.cache is None
        assert config.checkpoint is None

    def test_custom_values(self):
        """Test setting custom values."""
        config = BoltzPredictConfig(
            recycling_steps=5,
            sampling_steps=100,
            diffusion_samples=3,
            output_format="mmcif",
            write_full_pae=True,
            use_potentials=True,
            override=True,
            devices=4,
            accelerator="cpu",
            seed=42,
            use_msa_server=True,
            no_kernels=True,
            step_scale=1.5,
        )
        assert config.recycling_steps == 5
        assert config.sampling_steps == 100
        assert config.diffusion_samples == 3
        assert config.output_format == "mmcif"
        assert config.write_full_pae is True
        assert config.use_potentials is True
        assert config.override is True
        assert config.devices == 4
        assert config.accelerator == "cpu"
        assert config.seed == 42
        assert config.use_msa_server is True
        assert config.no_kernels is True
        assert config.step_scale == pytest.approx(1.5)


class TestBuildCommand:
    """Tests for build_command function."""

    def test_minimal_command(self, tmp_path):
        """Test building a minimal command with default config."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        config = BoltzPredictConfig()

        cmd = build_command(input_dir, output_dir, config)

        assert "predict" in cmd
        assert str(input_dir) in cmd
        assert str(output_dir) in cmd
        assert "--recycling_steps" in cmd
        assert "3" in cmd
        assert "--sampling_steps" in cmd
        assert "200" in cmd
        assert "--diffusion_samples" in cmd
        assert "1" in cmd
        assert "--output_format" in cmd
        assert "pdb" in cmd
        assert "--devices" in cmd
        assert "1" in cmd
        assert "--accelerator" in cmd
        assert "gpu" in cmd

    def test_command_with_all_options(self, tmp_path):
        """Test building a command with all options enabled."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"
        checkpoint = tmp_path / "model.ckpt"

        config = BoltzPredictConfig(
            cache=cache_dir,
            checkpoint=checkpoint,
            recycling_steps=5,
            sampling_steps=100,
            diffusion_samples=3,
            output_format="mmcif",
            write_full_pae=True,
            use_potentials=True,
            override=True,
            devices=4,
            accelerator="cpu",
            seed=42,
            use_msa_server=True,
            no_kernels=True,
            step_scale=1.5,
        )

        cmd = build_command(input_dir, output_dir, config)

        assert "--cache" in cmd
        assert str(cache_dir) in cmd
        assert "--checkpoint" in cmd
        assert str(checkpoint) in cmd
        assert "--seed" in cmd
        assert "42" in cmd
        assert "--write_full_pae" in cmd
        assert "--use_potentials" in cmd
        assert "--override" in cmd
        assert "--use_msa_server" in cmd
        assert "--no_kernels" in cmd
        assert "--step_scale" in cmd
        assert "1.5" in cmd


class TestPredictionsDir:
    """Tests for predictions_dir function."""

    def test_predictions_dir_path(self, tmp_path):
        """Test that predictions_dir returns correct path."""
        output_dir = tmp_path / "output"
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        result = predictions_dir(output_dir, input_dir)

        assert result == output_dir / f"boltz_results_{input_dir.stem}" / "predictions"
        assert result.name == "predictions"
        assert "boltz_results_inputs" in str(result)


class TestRenameModelOutputs:
    """Tests for rename_model_outputs function."""

    def test_rename_without_seed(self, tmp_path):
        """Test renaming model outputs without seed."""
        preds_dir = tmp_path / "predictions"
        preds_dir.mkdir()

        # Create test files
        (preds_dir / "entry_model_0.pdb").touch()
        (preds_dir / "entry_model_1.pdb").touch()
        (preds_dir / "entry_model_2.json").touch()

        rename_model_outputs(preds_dir, seed=None)

        # Check renamed files
        assert (preds_dir / "entry_sample0.pdb").exists()
        assert (preds_dir / "entry_sample1.pdb").exists()
        assert (preds_dir / "entry_sample2.json").exists()
        assert not (preds_dir / "entry_model_0.pdb").exists()

    def test_rename_with_seed(self, tmp_path):
        """Test renaming model outputs with seed."""
        preds_dir = tmp_path / "predictions"
        preds_dir.mkdir()

        # Create test files
        (preds_dir / "entry_model_0.pdb").touch()
        (preds_dir / "entry_model_1.mmcif").touch()

        rename_model_outputs(preds_dir, seed=42)

        # Check renamed files
        assert (preds_dir / "entry_seed42_sample0.pdb").exists()
        assert (preds_dir / "entry_seed42_sample1.mmcif").exists()
        assert not (preds_dir / "entry_model_0.pdb").exists()

    def test_rename_nested_files(self, tmp_path):
        """Test renaming works with nested directories."""
        preds_dir = tmp_path / "predictions"
        subdir = preds_dir / "subdir"
        subdir.mkdir(parents=True)

        # Create test files in subdirectory
        (subdir / "entry_model_0.pdb").touch()

        rename_model_outputs(preds_dir, seed=123)

        # Check renamed file
        assert (subdir / "entry_seed123_sample0.pdb").exists()
        assert not (subdir / "entry_model_0.pdb").exists()


class TestFmtDuration:
    """Tests for fmt_duration function."""

    def test_seconds(self):
        """Test formatting seconds."""
        assert fmt_duration(5.5) == "5.5s"
        assert fmt_duration(45.123) == "45.1s"

    def test_minutes(self):
        """Test formatting minutes."""
        assert fmt_duration(90) == "1m 30s"
        assert fmt_duration(150) == "2m 30s"

    def test_hours(self):
        """Test formatting hours."""
        assert fmt_duration(3600) == "1h 00m 00s"
        assert fmt_duration(3661) == "1h 01m 01s"
        assert fmt_duration(7200) == "2h 00m 00s"

    def test_hours_minutes_seconds(self):
        """Test formatting with all components."""
        assert fmt_duration(3723) == "1h 02m 03s"  # 1h 2m 3s


class TestPreprocessYamlsForMsa:
    """Tests for preprocess_yamls_for_msa function."""

    def test_remove_empty_msa(self, tmp_path):
        """Test removing 'msa: empty' from YAMLs."""
        import yaml

        from confidock.boltz.inference.runner import preprocess_yamls_for_msa

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        staging_dir = tmp_path / "staging"

        # Create a YAML with msa: empty
        yaml_data = {
            "sequences": [
                {"proteinA": {"sequence": "ACDEF", "msa": "empty"}},
                {"proteinB": {"sequence": "GHIKL", "msa": "empty"}},
            ]
        }
        yaml_path = input_dir / "test.yaml"
        with yaml_path.open("w") as fh:
            yaml.dump(yaml_data, fh)

        result = preprocess_yamls_for_msa(input_dir, staging_dir)

        assert result == staging_dir
        assert (staging_dir / "test.yaml").exists()

        # Check that msa: empty was removed
        with (staging_dir / "test.yaml").open() as fh:
            processed = yaml.safe_load(fh)

        assert "msa" not in processed["sequences"][0]["proteinA"]
        assert "msa" not in processed["sequences"][1]["proteinB"]
        assert processed["sequences"][0]["proteinA"]["sequence"] == "ACDEF"

    def test_preserve_non_empty_msa(self, tmp_path):
        """Test that non-empty MSA values are preserved."""
        import yaml

        from confidock.boltz.inference.runner import preprocess_yamls_for_msa

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        staging_dir = tmp_path / "staging"

        yaml_data = {
            "sequences": [
                {"proteinA": {"sequence": "ACDEF", "msa": "/path/to/msa.a3m"}},
            ]
        }
        yaml_path = input_dir / "test.yaml"
        with yaml_path.open("w") as fh:
            yaml.dump(yaml_data, fh)

        preprocess_yamls_for_msa(input_dir, staging_dir)

        with (staging_dir / "test.yaml").open() as fh:
            processed = yaml.safe_load(fh)

        assert processed["sequences"][0]["proteinA"]["msa"] == "/path/to/msa.a3m"

    def test_multiple_yaml_files(self, tmp_path):
        """Test processing multiple YAML files."""
        import yaml

        from confidock.boltz.inference.runner import preprocess_yamls_for_msa

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        staging_dir = tmp_path / "staging"

        for i in range(3):
            yaml_data = {"sequences": [{"protein": {"sequence": "ACDEF", "msa": "empty"}}]}
            yaml_path = input_dir / f"test{i}.yaml"
            with yaml_path.open("w") as fh:
                yaml.dump(yaml_data, fh)

        preprocess_yamls_for_msa(input_dir, staging_dir)

        assert (staging_dir / "test0.yaml").exists()
        assert (staging_dir / "test1.yaml").exists()
        assert (staging_dir / "test2.yaml").exists()

    def test_creates_staging_dir(self, tmp_path):
        """Test that staging directory is created."""
        from confidock.boltz.inference.runner import preprocess_yamls_for_msa

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        staging_dir = tmp_path / "nonexistent" / "staging"

        preprocess_yamls_for_msa(input_dir, staging_dir)

        assert staging_dir.exists()


class TestWritePredictionConfig:
    """Tests for write_prediction_config function."""

    def test_write_basic_config(self, tmp_path):
        """Test writing a basic prediction config."""
        import yaml

        from confidock.boltz.inference.runner import BoltzPredictConfig, write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        input_dir = tmp_path / "input"
        config = BoltzPredictConfig()

        result_path = write_prediction_config(output_dir, input_dir, config)

        assert result_path == output_dir / "prediction_config.yaml"
        assert result_path.exists()

        with result_path.open() as fh:
            data = yaml.safe_load(fh)

        assert "generated_at" in data
        assert data["dataset"] == "Unsal2026"
        assert "boltz2_settings" in data

    def test_config_with_msa_mode(self, tmp_path):
        """Test config with different MSA modes."""
        import yaml

        from confidock.boltz.inference.runner import BoltzPredictConfig, write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        input_dir = tmp_path / "input"
        config = BoltzPredictConfig()

        write_prediction_config(output_dir, input_dir, config, no_msa=True)

        cfg_path = output_dir / "prediction_config.yaml"
        with cfg_path.open() as fh:
            data = yaml.safe_load(fh)

        assert data["boltz2_settings"]["msa_mode"] == "single-sequence (no MSA)"

    def test_config_with_checkpoint(self, tmp_path):
        """Test config with custom checkpoint."""
        import yaml

        from confidock.boltz.inference.runner import BoltzPredictConfig, write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        input_dir = tmp_path / "input"
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.touch()

        config = BoltzPredictConfig(checkpoint=checkpoint)

        write_prediction_config(output_dir, input_dir, config)

        cfg_path = output_dir / "prediction_config.yaml"
        with cfg_path.open() as fh:
            data = yaml.safe_load(fh)

        assert str(checkpoint) in data["boltz2_settings"]["checkpoint"]

    def test_config_with_cache(self, tmp_path):
        """Test config with custom cache directory."""
        import yaml

        from confidock.boltz.inference.runner import BoltzPredictConfig, write_prediction_config

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        input_dir = tmp_path / "input"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        config = BoltzPredictConfig(cache=cache_dir)

        write_prediction_config(output_dir, input_dir, config)

        cfg_path = output_dir / "prediction_config.yaml"
        with cfg_path.open() as fh:
            data = yaml.safe_load(fh)

        assert str(cache_dir) in data["boltz2_settings"]["cache"]


class TestPrintTimings:
    """Tests for print_timings function."""

    def test_print_basic_timings(self, capsys):
        """Test printing basic timings."""
        from confidock.boltz.inference.runner import print_timings

        timings = {"Step 1": 10.5, "Step 2": 20.3}

        print_timings(timings)

        captured = capsys.readouterr()
        assert "Step 1" in captured.out
        assert "Step 2" in captured.out
        assert "Timing summary" in captured.out

    def test_print_with_inputs(self, capsys):
        """Test printing timings with input count."""
        from confidock.boltz.inference.runner import print_timings

        timings = {"Boltz-2 inference": 100.0}

        print_timings(timings, n_inputs=10, n_samples_per_input=2)

        captured = capsys.readouterr()
        assert "Avg per input" in captured.out
        assert "Avg per sample" in captured.out

    def test_print_with_zero_total(self, capsys):
        """Test printing timings with zero total time."""
        from confidock.boltz.inference.runner import print_timings

        timings = {"Step 1": 0.0}

        print_timings(timings)

        captured = capsys.readouterr()
        assert "Step 1" in captured.out


class TestBoltzExecutable:
    """Tests for boltz_executable function."""

    def test_returns_list(self):
        """Test that boltz_executable returns a list."""
        from confidock.boltz.inference.runner import boltz_executable

        result = boltz_executable()

        assert isinstance(result, list)
        assert len(result) > 0
