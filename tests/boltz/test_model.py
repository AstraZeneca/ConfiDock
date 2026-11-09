"""Unit tests for Boltz fine-tuning model wrapper."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from confidock.boltz import model as boltz_model
from confidock.boltz.dataset import BoltzFinetuneDataset
from confidock.boltz.model import BoltzFinetune


def _boltz_available() -> bool:
    try:
        import boltz  # noqa: F401
    except Exception:
        return False
    return True


def _load_model_module_without_boltz(monkeypatch: pytest.MonkeyPatch) -> object:
    """Re-import ``confidock.boltz.model`` as if the optional boltz dep were absent.

    ``None`` in ``sys.modules`` makes the import machinery raise, which is what drives
    :mod:`confidock.boltz.model` onto its stand-in base class.
    """
    for name in ("boltz", "boltz.model", "boltz.model.models.boltz2", "boltz.model.optim.scheduler"):
        monkeypatch.setitem(sys.modules, name, None)

    spec = importlib.util.spec_from_file_location("_model_without_boltz", boltz_model.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_is_callable_without_boltz(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BoltzFinetune(batch)`` must stay a call that reports the missing dependency.

    Without boltz installed the class is built on a stand-in base, and the tests below
    call the model directly.  If that base ever stops being callable, those calls die on
    a bare ``TypeError: 'BoltzFinetune' object is not callable`` -- true at runtime and
    flagged statically -- instead of naming the dependency that is missing.
    """
    module = _load_model_module_without_boltz(monkeypatch)
    model = module.BoltzFinetune()

    assert callable(model)
    with pytest.raises(RuntimeError, match="boltz is not installed"):
        model({"token_pad_mask": torch.ones(1, 2)})
    with pytest.raises(RuntimeError, match="boltz is not installed"):
        model.forward({"token_pad_mask": torch.ones(1, 2)})


def test_from_pretrained_without_boltz_names_the_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_model_module_without_boltz(monkeypatch)

    with pytest.raises(RuntimeError, match="cannot load pretrained checkpoint"):
        module.BoltzFinetune.from_pretrained("models/pretrained/boltz2.ckpt")


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    return Path("models/pretrained/boltz2.ckpt")


@pytest.fixture(scope="module")
def finetune_model(checkpoint_path: Path) -> BoltzFinetune:
    if not _boltz_available():
        pytest.skip("boltz package is not installed")
    if not checkpoint_path.exists():
        pytest.skip("pretrained checkpoint is not available")

    model = BoltzFinetune.from_pretrained(str(checkpoint_path))
    model.use_templates = False

    # Keep unit tests lightweight even when full training defaults are large.
    training_overrides = {
        "recycling_steps": 0,
        "num_sampling_steps": 1,
        "multiplicity_diffusion_train": 1,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
    }
    training_args = getattr(model, "training_args", None)
    for key, value in training_overrides.items():
        if isinstance(training_args, dict):
            training_args[key] = value
            continue
        try:
            training_args[key] = value
            continue
        except Exception:
            pass
        try:
            setattr(training_args, key, value)
        except Exception:
            pass

    return model


#: Boltz's atom encoder attends over fixed windows of this many atoms, so a batch
#: whose atom axis is not a multiple of it trips an internal reshape.
_ATOM_WINDOW = 32

#: Width of Boltz-2's ``res_type``/``profile`` token vocabulary.
_NUM_TOKENS = 33


def _build_dataset_item(tmp_path: Path, fixture: Path) -> dict[str, torch.Tensor | list[str]]:
    """Featurize ``fixture`` through the real dataset and return its single item."""
    structure_dir = tmp_path / "structures"
    processed_dir = tmp_path / "processed"
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(fixture, structure_dir / fixture.name)

    manifest = tmp_path / "train.csv"
    manifest.write_text(f"pdb_id,split,resolution\n{fixture.stem},train,1.8\n", encoding="utf-8")

    dataset = BoltzFinetuneDataset(
        structure_dir=str(structure_dir),
        processed_dir=str(processed_dir),
        split_manifest=str(manifest),
        max_tokens=512,
        max_atoms=1024,
    )
    return dataset[0]


def _truncate_atom_axes(
    item: dict[str, torch.Tensor | list[str]],
    *,
    atom_count: int,
    target: int,
) -> None:
    """Trim every tensor axis of length ``atom_count`` down to ``target``, in place.

    Length is the only thing identifying an atom axis here, so a token axis that
    happens to be exactly ``atom_count`` long would be trimmed too -- harmless for
    this fixture, where tokens far outnumber the 32-atom remainder.
    """
    for key, value in list(item.items()):
        if not torch.is_tensor(value):
            continue
        slices = [slice(None)] * value.ndim
        changed = False
        for dim, size in enumerate(value.shape):
            if size == atom_count:
                slices[dim] = slice(0, target)
                changed = True
        if changed:
            item[key] = value[tuple(slices)]


def _align_atom_axes(item: dict[str, torch.Tensor | list[str]]) -> None:
    """Round the atom axis down to a multiple of :data:`_ATOM_WINDOW`, in place."""
    atom_count = int(item["ref_pos"].shape[1])
    aligned_atom_count = atom_count - (atom_count % _ATOM_WINDOW)
    if 0 < aligned_atom_count < atom_count:
        _truncate_atom_axes(item, atom_count=atom_count, target=aligned_atom_count)


def _add_token_type_features(item: dict[str, torch.Tensor | list[str]], l_tokens: int) -> None:
    """Fill in ``res_type`` and ``profile`` if the dataset did not emit them."""
    if "res_type" not in item:
        res_type = torch.zeros((1, l_tokens, _NUM_TOKENS), dtype=torch.float32)
        res_type[:, :, 0] = 1.0
        item["res_type"] = res_type

    if "profile" not in item:
        profile = torch.zeros((1, l_tokens, _NUM_TOKENS), dtype=torch.float32)
        msa_profile = item.get("msa_profile")
        if torch.is_tensor(msa_profile):
            width = min(int(msa_profile.shape[-1]), _NUM_TOKENS)
            profile[:, :, :width] = msa_profile[:, :, :width].to(dtype=torch.float32)
        item["profile"] = profile


def _add_scalar_token_features(item: dict[str, torch.Tensor | list[str]], l_tokens: int) -> None:
    """Fill in the per-token scalars Boltz reads unconditionally, as zeros."""
    for key, dtype in (
        ("method_feature", torch.int64),
        ("modified", torch.int64),
        ("cyclic_period", torch.float32),
    ):
        if key not in item:
            item[key] = torch.zeros((1, l_tokens), dtype=dtype)


def _add_bond_and_msa_features(item: dict[str, torch.Tensor | list[str]], l_tokens: int) -> None:
    """Fill in ``type_bonds`` and clamp ``msa`` into the token vocabulary."""
    if "type_bonds" not in item:
        token_bonds = item.get("token_bonds")
        if torch.is_tensor(token_bonds):
            item["type_bonds"] = torch.zeros_like(token_bonds, dtype=torch.int64)
        else:
            item["type_bonds"] = torch.zeros((1, l_tokens, l_tokens), dtype=torch.int64)

    msa = item.get("msa")
    if torch.is_tensor(msa):
        item["msa"] = torch.clamp(msa.to(dtype=torch.int64), min=0, max=_NUM_TOKENS - 1)


def _add_msa_track_features(item: dict[str, torch.Tensor | list[str]], l_tokens: int) -> None:
    """Fill in the per-MSA-row tracks Boltz's MSA module reads, as zeros.

    ``BoltzFinetuneDataset`` builds a single-sequence MSA from the structure itself, so
    there are no deletions and no paired rows to describe -- but ``MSAModule.forward``
    indexes all three unconditionally, so their absence is a ``KeyError`` rather than a
    disabled feature.
    """
    msa = item.get("msa")
    n_rows = int(msa.shape[1]) if torch.is_tensor(msa) else 1
    for key in ("has_deletion", "deletion_value", "msa_paired"):
        if key not in item:
            item[key] = torch.zeros((1, n_rows, l_tokens), dtype=torch.float32)


def _add_missing_model_features(item: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor | list[str]]:
    """Top a dataset item up to what ``Boltz2.forward`` reads, in place."""
    l_tokens = int(item["token_pad_mask"].shape[1])
    _add_token_type_features(item, l_tokens)
    _add_scalar_token_features(item, l_tokens)
    _add_bond_and_msa_features(item, l_tokens)
    _add_msa_track_features(item, l_tokens)
    return item


@pytest.fixture(scope="module")
def model_batch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, torch.Tensor | list[str]]:
    fixture = Path("data/Unsal2026/models/PDBs/8b8i_H_A.pdb")
    if not fixture.exists():
        pytest.skip("real PDB fixture is not available")

    item = _build_dataset_item(tmp_path_factory.mktemp("boltz_model_batch"), fixture)
    _align_atom_axes(item)

    return _add_missing_model_features(item)


#: Atoms per attention window for the tiny model.  The queries window has to divide the
#: atom count -- boltz reshapes the atom axis into ``N // W`` windows -- which the real
#: model's 32 cannot do for a 12-atom structure, and 4 keeps the real 4:1 keys:queries
#: ratio.  Verified empirically: 5/16 and 4/7 both fail, 4/16 and 6/24 both work.
_TINY_ATOMS_PER_WINDOW_QUERIES = 4
_TINY_ATOMS_PER_WINDOW_KEYS = 16

#: Width of the reference-atom features ``BoltzFinetuneDataset`` emits: element one-hot
#: (128) + atom-name characters (4x64) + position (3) + charge (1).
_ATOM_FEATURE_DIM = 388

#: Trunk distogram bins.  Not a free parameter: ``ConfidenceHeads.forward`` multiplies the
#: distogram probabilities by a contact mask it allocates 64 wide.
_NUM_DISTOGRAM_BINS = 64


def _tiny_model_args() -> dict[str, object]:
    """Arguments for a randomly initialised, CPU-sized :class:`BoltzFinetune`.

    Boltz-2 takes its module shapes as arguments rather than reading them from a config,
    so a model small enough to build and run in a test has to be spelled out.  The four
    argument groups boltz itself publishes defaults for are taken from those dataclasses
    and only shrunk, so a boltz release that changes one of them (``v2`` on the pairformer
    was such a change) reaches this model instead of being pinned to a stale copy.
    """
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )

    pairformer = asdict(PairformerArgsV2())
    pairformer.update(num_blocks=1, num_heads=2)

    msa = asdict(MSAModuleArgs())
    msa.update(msa_s=8, msa_blocks=1, pairwise_head_width=8, pairwise_num_heads=2)

    diffusion = asdict(Boltz2DiffusionParams())
    # The trained model's reverse-diffusion alignment does an SVD on the sampled
    # coordinates; a randomly initialised one feeds it values it cannot decompose.
    diffusion.update(alignment_reverse_diff=False)

    # Every steering mode off: they are guidance on top of sampling, and each one reads
    # further features that a structure-only fixture has no way to supply.
    steering = asdict(BoltzSteeringParams())
    steering.update(fk_steering=False, physical_guidance_update=False, contact_guidance_update=False)

    training_args = {
        "recycling_steps": 0,
        "sampling_steps": 1,
        "diffusion_multiplicity": 1,
        "diffusion_samples": 1,
        "confidence_loss_weight": 1.0,
        "diffusion_loss_weight": 0.0,
        "distogram_loss_weight": 0.1,
        "bfactor_loss_weight": 0.0,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.95,
        "adam_eps": 1e-8,
        "lr_scheduler": "af3",
        "base_lr": 1e-4,
        "max_lr": 1e-4,
        "lr_warmup_no_steps": 1,
        "lr_start_decay_after_n_steps": 1,
        "lr_decay_every_n_steps": 1,
        "lr_decay_factor": 0.95,
        "weight_decay": 0.0,
        "symmetry_correction": False,
    }

    return {
        "atom_s": 32,
        "atom_z": 8,
        "token_s": 32,
        "token_z": 16,
        "num_bins": _NUM_DISTOGRAM_BINS,
        "atom_feature_dim": _ATOM_FEATURE_DIM,
        "training_args": training_args,
        "validation_args": {**training_args, "run_confidence_sequentially": True},
        "embedder_args": {
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 2,
            "add_mol_type_feat": True,
            "add_method_conditioning": True,
            "add_modified_flag": True,
            "add_cyclic_flag": True,
        },
        "msa_args": msa,
        "pairformer_args": pairformer,
        "score_model_args": {
            "sigma_data": 16,
            "dim_fourier": 8,
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 2,
            "token_transformer_depth": 1,
            "token_transformer_heads": 2,
            "atom_decoder_depth": 1,
            "atom_decoder_heads": 2,
            "conditioning_transition_layers": 1,
            "transformer_post_ln": False,
            "activation_checkpointing": False,
        },
        "diffusion_process_args": diffusion,
        "diffusion_loss_args": {
            "add_smooth_lddt_loss": True,
            "add_bond_loss": False,
            "nucleotide_loss_weight": 5.0,
            "ligand_loss_weight": 10.0,
        },
        "confidence_model_args": {
            "num_dist_bins": _NUM_DISTOGRAM_BINS,
            "max_dist": 22,
            "add_s_to_z_prod": True,
            "add_s_input_to_s": True,
            "add_z_input_to_z": True,
            "use_s_diffusion": False,
            "pairformer_args": {"num_blocks": 1, "num_heads": 2, "dropout": 0.0},
            "confidence_args": {
                "num_plddt_bins": 4,
                "num_pde_bins": 8,
                "num_pae_bins": 8,
                "use_separate_heads": True,
            },
        },
        "steering_args": steering,
        # The fine-tuning configuration itself: score a structure, train the confidence
        # heads only.  from_pretrained forces the same set onto a real checkpoint.
        "confidence_prediction": True,
        "affinity_prediction": False,
        "token_level_confidence": True,
        "structure_prediction_training": False,
        "run_trunk_and_structure": True,
        "skip_run_structure": False,
        "atoms_per_window_queries": _TINY_ATOMS_PER_WINDOW_QUERIES,
        "atoms_per_window_keys": _TINY_ATOMS_PER_WINDOW_KEYS,
        "use_templates": False,
        "predict_bfactor": False,
        "bond_type_feature": False,
        "checkpoint_diffusion_conditioning": False,
        "ema": False,
        "num_val_datasets": 1,
    }


@pytest.fixture(scope="module")
def tiny_finetune_model() -> BoltzFinetune:
    """A BoltzFinetune small enough to build and call without the pretrained checkpoint.

    The weights are random, so nothing here says anything about prediction quality.  What
    it does exercise is the wiring: that ``BoltzFinetune`` is constructible and callable,
    and that a confidence-configured forward pass still emits the outputs the fine-tune
    reads.  That is the part a boltz upgrade breaks, and it used to be checkable only by
    downloading 2.3 GB of weights -- so in practice it was not checked at all.
    """
    if not _boltz_available():
        pytest.skip("boltz package is not installed")

    torch.manual_seed(0)
    return BoltzFinetune(**_tiny_model_args())


@pytest.fixture(scope="module")
def tiny_model_batch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, torch.Tensor | list[str]]:
    """The in-repo fixture structure featurized for :func:`tiny_finetune_model`."""
    if not _boltz_available():
        pytest.skip("boltz package is not installed")

    fixture = Path("tests/boltz/fixtures/test_structure.pdb")
    item = _build_dataset_item(tmp_path_factory.mktemp("tiny_model_batch"), fixture)

    # No atom-axis alignment: the tiny model's windows are chosen to divide this
    # structure's atom count, so trimming it would only make the batch smaller.
    return _add_missing_model_features(item)


@pytest.fixture(
    params=[
        "tiny",
        # Same assertions against the real weights, when they happen to be present.
        pytest.param("pretrained", marks=pytest.mark.slow),
    ]
)
def forward_case(request: pytest.FixtureRequest) -> tuple[BoltzFinetune, dict[str, torch.Tensor | list[str]]]:
    """A (model, batch) pair to run a forward pass on, tiny or pretrained.

    The pretrained fixtures are requested lazily so that the tiny case does not pay for
    -- or get skipped by -- a checkpoint it does not need.
    """
    if request.param == "tiny":
        return request.getfixturevalue("tiny_finetune_model"), request.getfixturevalue("tiny_model_batch")
    return request.getfixturevalue("finetune_model"), request.getfixturevalue("model_batch")


@pytest.mark.slow
def test_from_pretrained_loads(finetune_model: BoltzFinetune) -> None:
    assert finetune_model is not None


@pytest.mark.slow
def test_only_confidence_parameters_trainable(finetune_model: BoltzFinetune) -> None:
    for name, param in finetune_model.named_parameters():
        if "confidence_module" in name or "out_token_feat_update" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_forward_pass_and_output_keys(
    forward_case: tuple[BoltzFinetune, dict[str, torch.Tensor | list[str]]],
) -> None:
    """A confidence-configured forward pass emits the outputs the fine-tune reads.

    Parametrized over a tiny random-init model and the pretrained one so the contract is
    actually executed: this used to depend on a 2.3 GB checkpoint, so the body never ran
    and neither the output keys nor the plain fact that ``BoltzFinetune`` is callable was
    ever checked -- the latter being a static complaint as well, since the class falls back
    to a stand-in base when boltz is missing.
    """
    model, batch = forward_case

    outputs = model(batch)

    for key in ["plddt", "pde", "ptm", "iptm", "pdistogram"]:
        assert key in outputs


@pytest.mark.slow
def test_training_step_returns_finite_scalar(
    finetune_model: BoltzFinetune,
    model_batch: dict[str, torch.Tensor | list[str]],
) -> None:
    loss = finetune_model.training_step(model_batch, batch_idx=0)

    if isinstance(loss, dict):
        loss = loss.get("loss", next(iter(loss.values())))

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


@pytest.mark.slow
def test_configure_optimizers_uses_confidence_params_only(finetune_model: BoltzFinetune) -> None:
    optim_config = finetune_model.configure_optimizers()

    optimizer = optim_config["optimizer"] if isinstance(optim_config, dict) else optim_config
    model_param_ids = {
        id(param)
        for name, param in finetune_model.named_parameters()
        if "confidence_module" in name or "out_token_feat_update" in name
    }

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            assert id(param) in model_param_ids
