"""Argument-parsing and entry-point tests for the workflow CLI scripts.

The scripts under ``workflows/scripts`` are thin wrappers: they parse flags and hand
off to ``confidock``, whose logic is unit-tested on its own. So these tests cover the
part the library tests cannot reach -- that each script's flags parse to the values
its ``main()`` expects, and that ``main()`` wires them through to the library calls --
with the library side mocked out.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

# Add workflows/scripts to path
WORKFLOWS_ROOT = Path(__file__).parent.parent / "workflows" / "scripts"
sys.path.insert(0, str(WORKFLOWS_ROOT / "processing"))
sys.path.insert(0, str(WORKFLOWS_ROOT / "inference"))


# ---------------------------------------------------------------------------
# prepare_boltz2_inputs.py
# ---------------------------------------------------------------------------

import prepare_boltz2_inputs  # noqa: E402


class TestPrepareBoltz2InputsArgs:
    """Test argument parsing for prepare_boltz2_inputs.py."""

    def test_parse_minimal_required_args(self):
        """Test parsing minimal required arguments."""
        with mock.patch("sys.argv", ["script", "--pdb_dir", "/pdbs", "--output_dir", "/out"]):
            args = prepare_boltz2_inputs._parse_args()
            assert args.pdb_dir == Path("/pdbs")
            assert args.output_dir == Path("/out")
            assert args.msa_dirs == []

    def test_parse_with_msa_dirs(self):
        """Test parsing multiple MSA directories."""
        with mock.patch(
            "sys.argv",
            ["script", "--pdb_dir", "/pdbs", "--output_dir", "/out", "--msa_dir", "/m1", "--msa_dir", "/m2"],
        ):
            args = prepare_boltz2_inputs._parse_args()
            assert len(args.msa_dirs) == 2

    def test_parse_with_filters(self):
        """Test parsing with date and sampling filters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--pdb_dir",
                "/pdbs",
                "--output_dir",
                "/out",
                "--cutoff_date",
                "2023-06-01",
                "--n_samples",
                "100",
                "--seed",
                "42",
                "--exclude_dir",
                "/exclude",
            ],
        ):
            args = prepare_boltz2_inputs._parse_args()
            assert args.cutoff_date == "2023-06-01"
            assert args.n_samples == 100
            assert args.seed == 42


@mock.patch("prepare_boltz2_inputs._generate_yamls_with_msas")
@mock.patch("prepare_boltz2_inputs._collect_pdb_files")
@mock.patch("sys.argv", ["script", "--pdb_dir", "/pdbs", "--output_dir", "/out"])
def test_prepare_boltz2_inputs_main(mock_collect, mock_generate, tmp_path):
    """Smoke test for main() function."""
    mock_collect.return_value = [tmp_path / "test.pdb"]
    mock_generate.return_value = (1, 0, 0)  # 1 success, 0 failures, 0 patches

    prepare_boltz2_inputs.main()

    assert mock_collect.called
    assert mock_generate.called


# ---------------------------------------------------------------------------
# run_boltz2_predictions.py
# ---------------------------------------------------------------------------

import run_boltz2_predictions  # noqa: E402


class TestRunBoltz2PredictionsArgs:
    """Test argument parsing for run_boltz2_predictions.py."""

    def test_parse_minimal_args(self):
        """Test parsing minimal arguments."""
        with mock.patch("sys.argv", ["script", "--input_dir", "/in", "--output_dir", "/out"]):
            args = run_boltz2_predictions._parse_args()
            assert args.input_dir == Path("/in")
            assert args.output_dir == Path("/out")

    def test_parse_with_cache_and_checkpoint(self):
        """Test parsing cache and checkpoint options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--cache",
                "/cache",
                "--checkpoint",
                "/ckpt.pt",
            ],
        ):
            args = run_boltz2_predictions._parse_args()
            assert args.cache == Path("/cache")
            assert args.checkpoint == Path("/ckpt.pt")

    def test_parse_sampling_params(self):
        """Test parsing sampling parameters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--recycling_steps",
                "5",
                "--sampling_steps",
                "200",
                "--diffusion_samples",
                "3",
                "--seed",
                "42",
            ],
        ):
            args = run_boltz2_predictions._parse_args()
            assert args.recycling_steps == 5
            assert args.sampling_steps == 200
            assert args.diffusion_samples == 3
            assert args.seed == 42

    def test_parse_flags(self):
        """Test boolean flags."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--write_full_pae",
                "--use_potentials",
                "--override",
                "--no_msa",
                "--no_kernels",
            ],
        ):
            args = run_boltz2_predictions._parse_args()
            assert args.write_full_pae is True
            assert args.use_potentials is True
            assert args.override is True
            assert args.no_msa is True
            assert args.no_kernels is True


@mock.patch("subprocess.run")
@mock.patch("run_boltz2_predictions.preprocess_yamls_for_msa")
@mock.patch("sys.argv", ["script", "--input_dir", "/in", "--output_dir", "/out"])
def test_run_boltz2_predictions_main(mock_preprocess, mock_subprocess, tmp_path, monkeypatch):
    """Smoke test for main() function."""
    # Create fake input directory with YAML files
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "test.yaml").write_text("sequences: []")

    output_dir = tmp_path / "out"

    monkeypatch.setattr("sys.argv", ["script", "--input_dir", str(input_dir), "--output_dir", str(output_dir)])

    mock_preprocess.return_value = input_dir
    mock_subprocess.return_value = mock.Mock(returncode=0)

    run_boltz2_predictions.main()

    assert mock_preprocess.called
    assert mock_subprocess.called


# ---------------------------------------------------------------------------
# run_boltz2_grid.py
# ---------------------------------------------------------------------------

import run_boltz2_grid  # noqa: E402


class TestRunBoltz2GridArgs:
    """Test argument parsing for run_boltz2_grid.py."""

    def test_parse_with_input_dir(self):
        """Test parsing with input_dir."""
        with mock.patch("sys.argv", ["script", "--input_dir", "/in", "--output_dir", "/out"]):
            args = run_boltz2_grid._parse_args()
            assert args.input_dir == Path("/in")
            assert args.output_dir == Path("/out")

    def test_parse_with_input_yamls(self):
        """Test parsing with explicit YAML files."""
        with mock.patch(
            "sys.argv",
            ["script", "--input_yamls", "/a.yaml", "/b.yaml", "--output_dir", "/out"],
        ):
            args = run_boltz2_grid._parse_args()
            assert len(args.input_yamls) == 2

    def test_parse_grid_params(self):
        """Test parsing grid parameters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--n_inputs",
                "10",
                "--seed",
                "42",
                "--diffusion_samples",
                "5",
            ],
        ):
            args = run_boltz2_grid._parse_args()
            assert args.n_inputs == 10
            assert args.seed == 42
            assert args.diffusion_samples == 5

    def test_parse_msa_modes(self):
        """Test parsing MSA modes."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--msa_modes",
                "full_msa",
                "no_msa",
            ],
        ):
            args = run_boltz2_grid._parse_args()
            assert "full_msa" in args.msa_modes
            assert "no_msa" in args.msa_modes

    def test_parse_flags(self):
        """Test parsing boolean flags."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_dir",
                "/in",
                "--output_dir",
                "/out",
                "--dry_run",
                "--overwrite",
                "--use_kernels",
            ],
        ):
            args = run_boltz2_grid._parse_args()
            assert args.dry_run is True
            assert args.overwrite is True
            assert args.use_kernels is True


@mock.patch("run_boltz2_grid._write_grid_config")
@mock.patch("run_boltz2_grid._collect_input_yamls")
@mock.patch("run_boltz2_grid.stage_yamls")
def test_run_boltz2_grid_main(mock_stage, mock_collect, mock_write_config, tmp_path):
    """Smoke test for main() with dry run."""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("sequences: []")
    mock_collect.return_value = [yaml_file]

    with mock.patch(
        "sys.argv", ["script", "--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out"), "--dry_run"]
    ):
        run_boltz2_grid.main()

    assert mock_collect.called
    assert mock_write_config.called


# ---------------------------------------------------------------------------
# run_esmfold2_predictions.py
# ---------------------------------------------------------------------------

import run_esmfold2_predictions  # noqa: E402


class TestRunEsmfold2PredictionsArgs:
    """Test argument parsing for run_esmfold2_predictions.py."""

    def test_parse_minimal_args(self):
        """Test parsing minimal arguments."""
        with mock.patch("sys.argv", ["script", "--input_json", "/in.json", "--output_dir", "/out"]):
            args = run_esmfold2_predictions._parse_args()
            assert args.input_json == Path("/in.json")
            assert args.output_dir == Path("/out")

    def test_parse_mode_options(self):
        """Test parsing mode options."""
        with mock.patch(
            "sys.argv",
            ["script", "--input_json", "/in.json", "--output_dir", "/out", "--mode", "forge"],
        ):
            args = run_esmfold2_predictions._parse_args()
            assert args.mode == "forge"

    def test_parse_sampling_params(self):
        """Test parsing sampling parameters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_json",
                "/in.json",
                "--output_dir",
                "/out",
                "--num_loops",
                "5",
                "--num_sampling_steps",
                "100",
                "--num_diffusion_samples",
                "3",
                "--seed",
                "42",
            ],
        ):
            args = run_esmfold2_predictions._parse_args()
            assert args.num_loops == 5
            assert args.num_sampling_steps == 100
            assert args.num_diffusion_samples == 3
            assert args.seed == 42

    def test_parse_msa_options(self):
        """Test parsing MSA options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_json",
                "/in.json",
                "--output_dir",
                "/out",
                "--msa_dir",
                "/msa",
                "--msa_max_sequences",
                "512",
            ],
        ):
            args = run_esmfold2_predictions._parse_args()
            assert args.msa_dir == Path("/msa")
            assert args.msa_max_sequences == 512


@mock.patch("run_esmfold2_predictions._run_local")
@mock.patch("run_esmfold2_predictions._load_entries")
def test_run_esmfold2_predictions_main(mock_load, mock_run, tmp_path):
    """Smoke test for main() in local mode."""
    input_json = tmp_path / "in.json"
    input_json.write_text('[{"name": "test", "sequences": []}]')
    output_dir = tmp_path / "out"

    mock_load.return_value = [{"name": "test", "sequences": []}]

    with mock.patch(
        "sys.argv", ["script", "--input_json", str(input_json), "--output_dir", str(output_dir), "--mode", "local"]
    ):
        run_esmfold2_predictions.main()

    assert mock_load.called
    assert mock_run.called


# ---------------------------------------------------------------------------
# run_esmfold2_grid.py
# ---------------------------------------------------------------------------

import run_esmfold2_grid  # noqa: E402


class TestRunEsmfold2GridArgs:
    """Test argument parsing for run_esmfold2_grid.py."""

    def test_parse_with_json_dir(self):
        """Test parsing with json_dir."""
        with mock.patch("sys.argv", ["script", "--json_dir", "/in", "--output_dir", "/out"]):
            args = run_esmfold2_grid._parse_args()
            assert args.json_dir == Path("/in")
            assert args.output_dir == Path("/out")

    def test_parse_with_input_jsons(self):
        """Test parsing with explicit JSON files."""
        with mock.patch(
            "sys.argv",
            ["script", "--input_jsons", "/a.json", "/b.json", "--output_dir", "/out"],
        ):
            args = run_esmfold2_grid._parse_args()
            assert len(args.input_jsons) == 2

    def test_parse_grid_params(self):
        """Test parsing grid parameters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--output_dir",
                "/out",
                "--n_inputs",
                "5",
                "--seed",
                "42",
                "--num_diffusion_samples",
                "2",
            ],
        ):
            args = run_esmfold2_grid._parse_args()
            assert args.n_inputs == 5
            assert args.seed == 42
            assert args.num_diffusion_samples == 2

    def test_parse_msa_modes(self):
        """Test parsing MSA modes."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--output_dir",
                "/out",
                "--msa_modes",
                "full_msa",
                "antigen_only",
            ],
        ):
            args = run_esmfold2_grid._parse_args()
            assert "full_msa" in args.msa_modes
            assert "antigen_only" in args.msa_modes

    def test_parse_local_mode_options(self):
        """Test parsing local mode options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--output_dir",
                "/out",
                "--mode",
                "local",
                "--local_model",
                "biohub/ESMFold2",
                "--device",
                "cuda:0",
            ],
        ):
            args = run_esmfold2_grid._parse_args()
            assert args.mode == "local"
            assert args.local_model == "biohub/ESMFold2"
            assert args.device == "cuda:0"

    def test_parse_forge_mode_options(self):
        """Test parsing forge mode options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--output_dir",
                "/out",
                "--mode",
                "forge",
                "--forge_model",
                "esmfold2-fast",
                "--forge_url",
                "https://api.example.com",
            ],
        ):
            args = run_esmfold2_grid._parse_args()
            assert args.mode == "forge"
            assert args.forge_model == "esmfold2-fast"
            assert args.forge_url == "https://api.example.com"


@mock.patch("run_esmfold2_grid._write_grid_config")
@mock.patch("run_esmfold2_grid._setup_inference_mode")
@mock.patch("run_esmfold2_grid._load_all_entries")
@mock.patch("run_esmfold2_grid._collect_input_files")
def test_run_esmfold2_grid_main(mock_collect, mock_load, mock_setup, mock_write_config, tmp_path):
    """Smoke test for main() with dry run."""
    json_file = tmp_path / "test.json"
    json_file.write_text('[{"name": "test", "sequences": []}]')
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()

    mock_collect.return_value = [json_file]
    mock_load.return_value = [{"name": "test", "sequences": []}]
    mock_setup.return_value = (None, None)

    with mock.patch(
        "sys.argv",
        [
            "script",
            "--json_dir",
            str(tmp_path),
            "--msa_dir",
            str(msa_dir),
            "--output_dir",
            str(tmp_path / "out"),
            "--dry_run",
            "--msa_modes",
            "no_msa",
        ],
    ):
        run_esmfold2_grid.main()

    assert mock_collect.called
    assert mock_load.called


# ---------------------------------------------------------------------------
# run_protenix_predictions.py
# ---------------------------------------------------------------------------

import run_protenix_predictions  # noqa: E402


class TestRunProtenixPredictionsArgs:
    """Test argument parsing for run_protenix_predictions.py."""

    def test_parse_minimal_args(self):
        """Test parsing minimal arguments."""
        with mock.patch("sys.argv", ["script", "--input_json", "/in.json", "--output_dir", "/out"]):
            args = run_protenix_predictions._parse_args()
            assert args.input_json == Path("/in.json")
            assert args.output_dir == Path("/out")

    def test_parse_model_and_seeds(self):
        """Test parsing model and seed options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_json",
                "/in.json",
                "--output_dir",
                "/out",
                "--model_name",
                "protenix-v2",
                "--seeds",
                "0,1,2",
                "--samples",
                "5",
            ],
        ):
            args = run_protenix_predictions._parse_args()
            assert args.model_name == "protenix-v2"
            assert args.seeds == "0,1,2"
            assert args.samples == 5

    def test_parse_sampling_params(self):
        """Test parsing sampling parameters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_json",
                "/in.json",
                "--output_dir",
                "/out",
                "--cycle",
                "10",
                "--step",
                "200",
            ],
        ):
            args = run_protenix_predictions._parse_args()
            assert args.cycle == 10
            assert args.step == 200

    def test_parse_msa_options(self):
        """Test parsing MSA options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_json",
                "/in.json",
                "--output_dir",
                "/out",
                "--use_msa",
                "--prefetch_msa",
                "--msa_dir",
                "/msa",
            ],
        ):
            args = run_protenix_predictions._parse_args()
            assert args.use_msa is True
            assert args.prefetch_msa is True
            assert args.msa_dir == Path("/msa")

    def test_parse_skip_existing(self):
        """Test parsing skip_existing flag."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_json",
                "/in.json",
                "--output_dir",
                "/out",
                "--skip_existing",
            ],
        ):
            args = run_protenix_predictions._parse_args()
            assert args.skip_existing is True


@mock.patch("subprocess.run")
@mock.patch("run_protenix_predictions._find_protenix_cli")
def test_run_protenix_predictions_main(mock_find_cli, mock_subprocess, tmp_path):
    """Smoke test for main() function."""
    input_json = tmp_path / "in.json"
    input_json.write_text('[{"name": "test", "sequences": []}]')
    output_dir = tmp_path / "out"

    mock_find_cli.return_value = ("protenix pred", "", "v2.0.0")
    mock_subprocess.return_value = mock.Mock(returncode=0)

    with mock.patch("sys.argv", ["script", "--input_json", str(input_json), "--output_dir", str(output_dir)]):
        run_protenix_predictions.main()

    assert mock_find_cli.called
    assert mock_subprocess.called


# ---------------------------------------------------------------------------
# run_protenix_grid.py
# ---------------------------------------------------------------------------

import run_protenix_grid  # noqa: E402


class TestRunProtenixGridArgs:
    """Test argument parsing for run_protenix_grid.py."""

    def test_parse_with_json_dir(self):
        """Test parsing with json_dir."""
        with mock.patch("sys.argv", ["script", "--json_dir", "/in", "--msa_dir", "/msa", "--output_dir", "/out"]):
            args = run_protenix_grid._parse_args()
            assert args.json_dir == Path("/in")
            assert args.msa_dir == Path("/msa")
            assert args.output_dir == Path("/out")

    def test_parse_with_input_jsons(self):
        """Test parsing with explicit JSON files."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--input_jsons",
                "/a.json",
                "/b.json",
                "--msa_dir",
                "/msa",
                "--output_dir",
                "/out",
            ],
        ):
            args = run_protenix_grid._parse_args()
            assert len(args.input_jsons) == 2

    def test_parse_grid_params(self):
        """Test parsing grid parameters."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--msa_dir",
                "/msa",
                "--output_dir",
                "/out",
                "--n_inputs",
                "5",
                "--seed",
                "42",
                "--samples",
                "3",
            ],
        ):
            args = run_protenix_grid._parse_args()
            assert args.n_inputs == 5
            assert args.seed == 42
            assert args.samples == 3

    def test_parse_model_options(self):
        """Test parsing model options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--msa_dir",
                "/msa",
                "--output_dir",
                "/out",
                "--model_name",
                "protenix-v2",
                "--checkpoint_dir",
                "/ckpt",
                "--seeds",
                "0,1,2",
            ],
        ):
            args = run_protenix_grid._parse_args()
            assert args.model_name == "protenix-v2"
            assert args.checkpoint_dir == Path("/ckpt")
            assert args.seeds == "0,1,2"

    def test_parse_msa_modes(self):
        """Test parsing MSA modes."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--msa_dir",
                "/msa",
                "--output_dir",
                "/out",
                "--msa_modes",
                "full_msa",
                "no_msa",
            ],
        ):
            args = run_protenix_grid._parse_args()
            assert "full_msa" in args.msa_modes
            assert "no_msa" in args.msa_modes

    def test_parse_ddp_options(self):
        """Test parsing DDP options."""
        with mock.patch(
            "sys.argv",
            [
                "script",
                "--json_dir",
                "/in",
                "--msa_dir",
                "/msa",
                "--output_dir",
                "/out",
                "--use_ddp",
                "--cuda_devices",
                "0,1,2,3",
                "--num_workers",
                "4",
            ],
        ):
            args = run_protenix_grid._parse_args()
            assert args.use_ddp is True
            assert args.cuda_devices == "0,1,2,3"
            assert args.num_workers == 4


@mock.patch("run_protenix_grid._find_protenix_cli")
@mock.patch("run_protenix_grid._load_protenix_entries")
@mock.patch("run_protenix_grid._collect_input_json_files")
def test_run_protenix_grid_main(mock_collect, mock_load, mock_find_cli, tmp_path):
    """Smoke test for main() with dry run."""
    mock_collect.return_value = [tmp_path / "test.json"]
    mock_load.return_value = [{"name": "test", "sequences": []}]
    mock_find_cli.return_value = ("protenix", "/path", "v1.0")

    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()

    with mock.patch(
        "sys.argv",
        [
            "script",
            "--json_dir",
            str(tmp_path),
            "--msa_dir",
            str(msa_dir),
            "--output_dir",
            str(tmp_path / "out"),
            "--dry_run",
        ],
    ):
        run_protenix_grid.main()

    assert mock_collect.called
    assert mock_load.called
