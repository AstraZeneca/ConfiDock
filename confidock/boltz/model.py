"""Model wrapper for Boltz confidence module fine-tuning."""

from __future__ import annotations

import argparse
import collections
import inspect
import logging
import pickle
import typing
from collections.abc import Mapping
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)

try:
    from boltz.model.models.boltz2 import Boltz2
    from boltz.model.optim.scheduler import AlphaFoldLRScheduler
except Exception:  # pragma: no cover - dependency availability is environment-specific

    class _Boltz2Unavailable:
        """Stand-in base class used when the optional ``boltz`` package is missing.

        This used to be plain ``object``, which made :class:`BoltzFinetune` a
        non-callable, non-``Module`` class in that environment: the import still
        succeeded, but calling an instance died on a bare ``TypeError: 'BoltzFinetune'
        object is not callable``, and static analysis of ``finetune_model(batch)``
        in the tests was right to flag a call on something not callable.  Naming the
        missing dependency here keeps the import side-effect free while making both
        the runtime error and the inferred type honest.
        """

        _MESSAGE = "boltz is not installed or incompatible; BoltzFinetune cannot be used"

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Fail with the missing-dependency reason instead of a bare TypeError."""
            raise RuntimeError(self._MESSAGE)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            """Fail with the missing-dependency reason instead of a bare TypeError."""
            raise RuntimeError(self._MESSAGE)

    Boltz2 = _Boltz2Unavailable  # type: ignore[assignment, misc]
    AlphaFoldLRScheduler = None


class BoltzFinetune(Boltz2):  # type: ignore[misc]
    """Boltz2 subclass configured for confidence module fine-tuning."""

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **override_kwargs: Any) -> "BoltzFinetune":
        """Load pretrained Boltz2 weights and reconfigure for confidence fine-tuning."""
        if not hasattr(super(), "load_from_checkpoint"):
            raise RuntimeError("boltz is not installed or incompatible; cannot load pretrained checkpoint")

        forced_overrides = {
            "structure_prediction_training": False,
            "confidence_prediction": True,
            "affinity_prediction": False,
            "run_trunk_and_structure": True,
            "skip_run_structure": False,
        }
        forced_overrides.update(override_kwargs)

        # Some checkpoints include diffusion kwargs that older/newer boltz package
        # versions do not accept. Filter those keys before model construction.
        compatible_diffusion_args = _extract_compatible_diffusion_process_args(checkpoint_path)
        if compatible_diffusion_args is not None:
            forced_overrides["diffusion_process_args"] = compatible_diffusion_args

        model = cls.load_from_checkpoint(  # type: ignore[attr-defined]
            checkpoint_path,
            strict=False,
            **forced_overrides,
        )
        _set_confidence_trainable_only(model)
        return model

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler for confidence-only fine-tuning."""
        params = [
            p
            for name, p in self.named_parameters()
            if ("confidence_module" in name or "out_token_feat_update" in name) and p.requires_grad
        ]

        if not params:
            raise RuntimeError("No trainable confidence parameters found for optimizer setup")

        optimizer = torch.optim.AdamW(
            params,
            lr=float(_get_arg(self.training_args, "base_lr", 1e-4)),
            betas=(
                float(_get_arg(self.training_args, "adam_beta_1", 0.9)),
                float(_get_arg(self.training_args, "adam_beta_2", 0.999)),
            ),
            eps=float(_get_arg(self.training_args, "adam_eps", 1e-8)),
            weight_decay=float(_get_arg(self.training_args, "weight_decay", 0.0)),
        )

        if AlphaFoldLRScheduler is None:
            return {"optimizer": optimizer}

        scheduler = _build_alphafold_scheduler(optimizer=optimizer, training_args=self.training_args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        """Set fine-tuning loss weights for this step and delegate to Boltz2."""
        updates = {
            "confidence_loss_weight": 1.0,
            "diffusion_loss_weight": 0.0,
            "distogram_loss_weight": 0.1,
            "bfactor_loss_weight": 0.0,
        }

        original = {key: _get_arg(self.training_args, key, None) for key in updates}
        for key, value in updates.items():
            _set_arg(self.training_args, key, value)

        try:
            return super().training_step(batch, batch_idx)
        finally:
            for key, value in original.items():
                _set_arg(self.training_args, key, value)


def _build_alphafold_scheduler(optimizer: torch.optim.Optimizer, training_args: Any) -> Any:
    if AlphaFoldLRScheduler is None:
        raise RuntimeError("AlphaFoldLRScheduler unavailable")

    scheduler_params = {
        "base_lr": float(_get_arg(training_args, "base_lr", 1e-4)),
        "max_lr": float(_get_arg(training_args, "max_lr", _get_arg(training_args, "base_lr", 1e-4))),
        "lr_warmup_no_steps": int(_get_arg(training_args, "lr_warmup_no_steps", 500)),
        "lr_start_decay_after_n_steps": int(_get_arg(training_args, "lr_start_decay_after_n_steps", 5000)),
        "lr_decay_every_n_steps": int(_get_arg(training_args, "lr_decay_every_n_steps", 50000)),
        "lr_decay_factor": float(_get_arg(training_args, "lr_decay_factor", 0.95)),
    }

    signature = inspect.signature(AlphaFoldLRScheduler)
    kwargs = {}
    for key, value in scheduler_params.items():
        if key in signature.parameters:
            kwargs[key] = value

    if "optimizer" in signature.parameters:
        kwargs["optimizer"] = optimizer

    return AlphaFoldLRScheduler(**kwargs)


def _get_arg(container: Any, key: str, default: Any) -> Any:
    if container is None:
        return default
    if hasattr(container, "get"):
        try:
            value = container.get(key, default)
            return default if value is None else value
        except Exception:
            pass
    if isinstance(container, dict):
        value = container.get(key, default)
        return default if value is None else value
    if hasattr(container, key):
        value = getattr(container, key)
        return default if value is None else value
    return default


def _set_arg(container: Any, key: str, value: Any) -> None:
    if container is None:
        return
    if isinstance(container, dict):
        container[key] = value
        return
    try:
        container[key] = value
        return
    except Exception:
        pass
    try:
        setattr(container, key, value)
    except Exception:
        pass


def _metadata_safe_globals() -> list[Any]:
    """List the container types a checkpoint's ``hyper_parameters`` is allowed to hold.

    Everything here is an inert data constructor -- builtin containers, the two
    ``collections`` dict flavours, and the omegaconf nodes a hydra-configured run
    pickles its config as.  Allowlisting exactly these is what lets the metadata read
    run under ``weights_only=True``: the unpickler will refuse any other global, so
    nothing in the file can be executed.
    """
    safe: list[Any] = [
        bool,
        bytes,
        complex,
        dict,
        float,
        int,
        list,
        set,
        str,
        tuple,
        argparse.Namespace,
        collections.OrderedDict,
        collections.defaultdict,
        typing.Any,
    ]
    try:
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode
    except Exception:  # pragma: no cover - omegaconf ships with boltz, but is optional here
        return safe
    return [*safe, AnyNode, ContainerMetadata, DictConfig, ListConfig, Metadata]


def _load_checkpoint_hyper_parameters(checkpoint_path: str) -> Any:
    """Read a checkpoint's ``hyper_parameters`` without executing code from the file.

    ``weights_only=True`` cannot read a Boltz-2 checkpoint on its own -- the
    hyper-parameters are omegaconf nodes, which the restricted unpickler rejects by
    default -- so the read runs with those container types allowlisted instead of with
    the restriction turned off.  The allowlist is scoped to this call rather than
    registered globally: it must not silently widen what any other ``torch.load`` in
    the process accepts.

    Returns an empty mapping when the checkpoint carries no readable metadata; the
    caller treats that as "nothing to filter with".
    """
    try:
        with torch.serialization.safe_globals(_metadata_safe_globals()):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        # A global the allowlist does not name.  There is deliberately no
        # ``weights_only=False`` retry: a hostile payload is exactly a global the
        # allowlist does not name, so a fallback would run it and leave the
        # restriction protecting only the files that never needed it.  A benign
        # checkpoint format that lands here loses the caller's diffusion-arg filtering
        # instead, which surfaces as an unexpected-kwarg TypeError from AtomDiffusion --
        # hence the warning: it names the rejected type, i.e. what to add here.
        LOGGER.warning("could not read hyper_parameters from %s under weights_only: %s", checkpoint_path, error)
        return {}

    if not isinstance(checkpoint, Mapping):
        return {}
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    # Older Lightning stored an argparse.Namespace, which has no .get; there is nothing
    # to read out of it here either way.
    return hyper_parameters if hasattr(hyper_parameters, "get") else {}


def _extract_compatible_diffusion_process_args(checkpoint_path: str) -> dict[str, Any] | None:
    """Return checkpoint diffusion args filtered to the local AtomDiffusion signature."""
    try:
        hyper_parameters = _load_checkpoint_hyper_parameters(checkpoint_path)
        diffusion_process_args = hyper_parameters.get("diffusion_process_args")
        if diffusion_process_args is None:
            return None

        if not isinstance(diffusion_process_args, dict):
            try:
                from omegaconf import OmegaConf

                diffusion_process_args = OmegaConf.to_container(diffusion_process_args, resolve=True)
            except Exception:
                pass

        if isinstance(diffusion_process_args, Mapping):
            diffusion_process_args = dict(diffusion_process_args)
        if not isinstance(diffusion_process_args, dict):
            return None

        signature = _get_atom_diffusion_signature()
        if signature is None:
            return diffusion_process_args

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return diffusion_process_args

        allowed_keys = {
            name for name in signature.parameters if name not in {"self", "score_model_args", "compile_score"}
        }
        return {key: value for key, value in diffusion_process_args.items() if key in allowed_keys}
    except Exception:
        return None


def _get_atom_diffusion_signature() -> inspect.Signature | None:
    """Get signature for whichever AtomDiffusion implementation is installed."""
    candidate_imports = (
        "boltz.model.modules.diffusionv2",
        "boltz.model.modules.diffusion",
    )
    for module_name in candidate_imports:
        try:
            module = __import__(module_name, fromlist=["AtomDiffusion"])
            atom_diffusion = getattr(module, "AtomDiffusion", None)
            if atom_diffusion is not None:
                return inspect.signature(atom_diffusion.__init__)
        except Exception:
            continue
    return None


def _set_confidence_trainable_only(model: Any) -> None:
    """Freeze all parameters except confidence heads used for fine-tuning."""
    try:
        for _, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "confidence_module" in name or "out_token_feat_update" in name:
                param.requires_grad = True
    except Exception:
        pass
