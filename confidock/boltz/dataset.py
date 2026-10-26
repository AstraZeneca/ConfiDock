"""Dataset utilities for Boltz confidence module fine-tuning."""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from boltz.data import const as boltz_const
except Exception:  # pragma: no cover - dependency availability is environment-specific
    boltz_const = None

LOGGER = logging.getLogger(__name__)

REQUIRED_KEYS = {
    "coords",
    "atom_pad_mask",
    "atom_resolved_mask",
    "token_pad_mask",
    "token_index",
    "residue_index",
    "chain_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "mol_type",
    "token_bonds",
    "msa",
    "msa_mask",
    "msa_profile",
    "deletion_mean",
    "frames_idx",
    "frame_resolved_mask",
    "pdb_id",
    "idx_dataset",
}


@dataclass
class _ParsedFallbackStructure:
    """Flat atom and per-token records collected by the fallback PDB parser."""

    atom_coords: list[list[float]] = field(default_factory=list)
    atom_resolved: list[float] = field(default_factory=list)
    residue_index: list[int] = field(default_factory=list)
    chain_index: list[int] = field(default_factory=list)
    token_atom_indices: list[list[int]] = field(default_factory=list)


class BoltzFinetuneDataset(Dataset):
    """Dataset that loads and caches Boltz-ready features from PDB structures."""

    def __init__(
        self,
        structure_dir: str,
        processed_dir: str,
        split_manifest: str,
        max_tokens: int = 512,
        max_atoms: int = 4096,
    ) -> None:
        self.structure_dir = Path(structure_dir)
        self.processed_dir = Path(processed_dir)
        self.split_manifest = Path(split_manifest)
        self.max_tokens = int(max_tokens)
        self.max_atoms = int(max_atoms)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._entries = self._load_entries(self.split_manifest)

    def _load_entries(self, manifest_path: Path) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        if not manifest_path.exists():
            LOGGER.warning("Split manifest does not exist: %s", manifest_path)
            return entries

        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                LOGGER.warning("Manifest has no header: %s", manifest_path)
                return entries

            for row in reader:
                pdb_id = (row.get("pdb_id") or "").strip()
                if not pdb_id:
                    continue
                pdb_path = self.structure_dir / f"{pdb_id}.pdb"
                if not pdb_path.exists():
                    LOGGER.warning("Missing PDB for %s at %s; skipping entry.", pdb_id, pdb_path)
                    continue
                entries.append(row)

        return entries

    def __len__(self) -> int:
        """Return the number of entries in the dataset."""
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a dataset item by index."""
        if index < 0 or index >= len(self._entries):
            raise IndexError("Dataset index out of range")

        entry = self._entries[index]
        pdb_id = str(entry["pdb_id"])
        cache_file = self.processed_dir / f"{pdb_id}.pt"

        if cache_file.exists():
            feats = torch.load(cache_file, map_location="cpu")
            if not isinstance(feats, dict):
                raise TypeError(f"Cached features at {cache_file} are not a dictionary")
            _, changed = self._ensure_model_features(feats)
            if changed:
                torch.save(feats, cache_file)
            return feats

        pdb_path = self.structure_dir / f"{pdb_id}.pdb"

        try:
            feats = self._featurize_with_boltz(pdb_path=pdb_path, pdb_id=pdb_id)
            if feats is None:
                feats = self._featurize_fallback(pdb_path=pdb_path, pdb_id=pdb_id)
        except Exception as exc:  # pragma: no cover - defensive runtime safeguard
            LOGGER.warning("Failed to parse %s: %s. Skipping entry.", pdb_path, exc)
            if index + 1 < len(self):
                return self[index + 1]
            raise IndexError("No valid entries are available in this dataset") from exc

        feats, _ = self._ensure_model_features(feats)
        torch.save(feats, cache_file)
        return feats

    def _featurize_with_boltz(self, pdb_path: Path, pdb_id: str) -> dict[str, Any] | None:
        """Try to featurize via Boltz internals and coerce into required keys."""
        try:
            import boltz.data.feature as boltz_feature
            import boltz.data.parse as boltz_parse
        except Exception:
            return None

        parsed_obj = self._call_first_available(
            boltz_parse,
            ("parse_pdb", "parse_structure", "load_structure"),
            lambda parser: parser(str(pdb_path)),
            lambda parser: parser(pdb_path),
        )
        if parsed_obj is None:
            return None

        raw_features = self._call_first_available(
            boltz_feature,
            ("featurize", "create_features", "build_features", "structure_to_features"),
            lambda featurizer: featurizer(parsed_obj, max_tokens=self.max_tokens, max_atoms=self.max_atoms),
            lambda featurizer: featurizer(parsed_obj),
        )
        if not isinstance(raw_features, dict):
            return None

        return self._coerce_feature_dict(raw_features=raw_features, pdb_id=pdb_id)

    @staticmethod
    def _call_first_available(
        module: Any,
        names: tuple[str, ...],
        call: Callable[[Any], Any],
        call_without_kwargs: Callable[[Any], Any],
    ) -> Any | None:
        """Call the first attribute of ``module`` in ``names`` that succeeds.

        ``call`` is attempted first; if the attribute rejects those arguments
        (``TypeError``), ``call_without_kwargs`` is tried before moving on to the
        next candidate name.
        """
        for name in names:
            func = getattr(module, name, None)
            if func is None:
                continue
            try:
                return call(func)
            except TypeError:
                pass
            except Exception:
                continue
            try:
                return call_without_kwargs(func)
            except Exception:
                continue
        return None

    def _coerce_feature_dict(self, raw_features: dict[str, Any], pdb_id: str) -> dict[str, Any] | None:
        if not REQUIRED_KEYS.issubset(raw_features.keys() | {"pdb_id", "idx_dataset"}):
            return None

        dtype_map = {
            "coords": torch.float32,
            "atom_pad_mask": torch.float32,
            "atom_resolved_mask": torch.float32,
            "token_pad_mask": torch.float32,
            "token_index": torch.int64,
            "residue_index": torch.int64,
            "chain_index": torch.int64,
            "asym_id": torch.int64,
            "entity_id": torch.int64,
            "sym_id": torch.int64,
            "mol_type": torch.int64,
            "token_bonds": torch.float32,
            "msa": torch.int64,
            "msa_mask": torch.float32,
            "msa_profile": torch.float32,
            "deletion_mean": torch.float32,
            "frames_idx": torch.int64,
            "frame_resolved_mask": torch.float32,
            "idx_dataset": torch.int64,
        }

        dims_map = {
            "coords": 4,
            "atom_pad_mask": 2,
            "atom_resolved_mask": 2,
            "token_pad_mask": 2,
            "token_index": 2,
            "residue_index": 2,
            "chain_index": 2,
            "asym_id": 2,
            "entity_id": 2,
            "sym_id": 2,
            "mol_type": 2,
            "token_bonds": 4,
            "msa": 3,
            "msa_mask": 3,
            "msa_profile": 3,
            "deletion_mean": 2,
            "frames_idx": 4,
            "frame_resolved_mask": 3,
            "idx_dataset": 1,
        }

        feats: dict[str, Any] = {}
        for key, dtype in dtype_map.items():
            value = raw_features.get(key)
            if value is None and key == "idx_dataset":
                value = torch.tensor([0], dtype=torch.int64)
            if value is None:
                return None
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            tensor = tensor.to(dtype=dtype)
            target_dims = dims_map[key]
            while tensor.dim() < target_dims:
                tensor = tensor.unsqueeze(0)
            feats[key] = tensor

        feats["pdb_id"] = [pdb_id]
        return feats

    def _ensure_model_features(
        self,
        feats: dict[str, Any],
        token_atom_indices: list[list[int]] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Add model-required features that may be absent from lightweight featurization.

        Each feature group is filled in by a dedicated helper; every helper mutates
        ``feats`` in place and reports whether it wrote a feature that was missing.
        """
        if not torch.is_tensor(feats.get("coords")) or not torch.is_tensor(feats.get("token_pad_mask")):
            return feats, False

        l_atoms = int(feats["coords"].shape[2])
        l_tokens = int(feats["token_pad_mask"].shape[1])

        changed = self._ensure_atom_to_token(
            feats=feats,
            l_atoms=l_atoms,
            l_tokens=l_tokens,
            token_atom_indices=token_atom_indices,
        )
        atom_to_token = feats["atom_to_token"]

        changed |= self._ensure_token_to_rep_atom(
            feats=feats,
            atom_to_token=atom_to_token,
            l_atoms=l_atoms,
            l_tokens=l_tokens,
        )
        changed |= self._ensure_atom_reference_features(feats=feats, atom_to_token=atom_to_token, l_atoms=l_atoms)
        changed |= self._ensure_ref_element(feats=feats, l_atoms=l_atoms)
        changed |= self._ensure_ref_atom_name_chars(feats=feats, l_atoms=l_atoms)
        changed |= self._ensure_token_features(feats=feats, l_tokens=l_tokens)
        changed |= self._ensure_contact_features(feats=feats, l_tokens=l_tokens)

        return feats, changed

    def _ensure_atom_to_token(
        self,
        feats: dict[str, Any],
        l_atoms: int,
        l_tokens: int,
        token_atom_indices: list[list[int]] | None,
    ) -> bool:
        """Add or repair the atom-to-token assignment matrix."""
        atom_to_token = feats.get("atom_to_token")
        if not torch.is_tensor(atom_to_token):
            feats["atom_to_token"] = self._build_atom_to_token(
                l_atoms=l_atoms,
                l_tokens=l_tokens,
                token_atom_indices=token_atom_indices,
            )
            return True

        atom_to_token = atom_to_token.to(dtype=torch.float32)
        while atom_to_token.dim() < 3:
            atom_to_token = atom_to_token.unsqueeze(0)
        if atom_to_token.shape != (1, l_atoms, l_tokens):
            atom_to_token = self._build_atom_to_token(
                l_atoms=l_atoms,
                l_tokens=l_tokens,
                token_atom_indices=token_atom_indices,
            )
        feats["atom_to_token"] = atom_to_token
        return False

    @staticmethod
    def _ensure_token_to_rep_atom(
        feats: dict[str, Any],
        atom_to_token: torch.Tensor,
        l_atoms: int,
        l_tokens: int,
    ) -> bool:
        """Pick one representative atom per token."""
        if "token_to_rep_atom" in feats:
            return False

        token_to_rep_atom = torch.zeros((1, l_tokens, l_atoms), dtype=torch.float32)
        token_ids = torch.argmax(atom_to_token[0], dim=-1)
        for token_id in range(l_tokens):
            atom_ids = torch.where(token_ids == token_id)[0]
            atom_index = int(atom_ids[0]) if atom_ids.numel() > 0 else min(token_id, l_atoms - 1)
            token_to_rep_atom[0, token_id, atom_index] = 1.0
        feats["token_to_rep_atom"] = token_to_rep_atom
        return True

    @staticmethod
    def _ensure_atom_reference_features(
        feats: dict[str, Any],
        atom_to_token: torch.Tensor,
        l_atoms: int,
    ) -> bool:
        """Fill in the per-atom reference-conformer features."""
        changed = False

        if "r_set_to_rep_atom" not in feats:
            feats["r_set_to_rep_atom"] = torch.eye(l_atoms, dtype=torch.float32).unsqueeze(0)
            changed = True

        if "ref_pos" not in feats:
            feats["ref_pos"] = feats["coords"][:, 0].to(dtype=torch.float32)
            changed = True

        if "ref_space_uid" not in feats:
            asym_id = feats.get("asym_id")
            if torch.is_tensor(asym_id):
                ref_space_uid = torch.bmm(atom_to_token, asym_id.unsqueeze(-1).float()).squeeze(-1).long()
            else:
                ref_space_uid = torch.zeros((1, l_atoms), dtype=torch.int64)
            feats["ref_space_uid"] = ref_space_uid
            changed = True

        if "ref_charge" not in feats:
            feats["ref_charge"] = torch.zeros((1, l_atoms), dtype=torch.float32)
            changed = True

        return changed

    @staticmethod
    def _ensure_ref_element(feats: dict[str, Any], l_atoms: int) -> bool:
        """Add or reshape the one-hot element identity per atom."""
        num_elements = int(getattr(boltz_const, "num_elements", 128))
        ref_element = feats.get("ref_element")

        if not torch.is_tensor(ref_element):
            ref_element = torch.zeros((1, l_atoms, num_elements), dtype=torch.float32)
            ref_element[:, :, 0] = 1.0
            feats["ref_element"] = ref_element
            return True

        changed = False
        ref_element = ref_element.to(dtype=torch.float32)
        while ref_element.dim() < 3:
            ref_element = ref_element.unsqueeze(-1)
        if ref_element.shape[0] != 1 or ref_element.shape[1] != l_atoms or ref_element.shape[-1] != num_elements:
            fixed = torch.zeros((1, l_atoms, num_elements), dtype=torch.float32)
            width = min(ref_element.shape[-1], num_elements)
            fixed[:, :, :width] = ref_element[:, :l_atoms, :width]
            ref_element = fixed
            changed = True
        feats["ref_element"] = ref_element
        return changed

    @staticmethod
    def _ensure_ref_atom_name_chars(feats: dict[str, Any], l_atoms: int) -> bool:
        """Add or reshape the one-hot atom-name characters, shaped (1, l_atoms, 4, 64)."""
        ref_atom_name_chars = feats.get("ref_atom_name_chars")

        if not torch.is_tensor(ref_atom_name_chars):
            ref_atom_name_chars = torch.zeros((1, l_atoms, 4, 64), dtype=torch.float32)
            ref_atom_name_chars[:, :, :, 0] = 1.0
            feats["ref_atom_name_chars"] = ref_atom_name_chars
            return True

        changed = False
        ref_atom_name_chars = ref_atom_name_chars.to(dtype=torch.float32)
        if ref_atom_name_chars.dim() == 3 and ref_atom_name_chars.shape[-1] == 256:
            ref_atom_name_chars = ref_atom_name_chars.reshape(1, l_atoms, 4, 64)
            changed = True
        while ref_atom_name_chars.dim() < 4:
            ref_atom_name_chars = ref_atom_name_chars.unsqueeze(-1)
        if (
            ref_atom_name_chars.shape[0] != 1
            or ref_atom_name_chars.shape[1] != l_atoms
            or ref_atom_name_chars.shape[2] != 4
            or ref_atom_name_chars.shape[3] != 64
        ):
            fixed_chars = torch.zeros((1, l_atoms, 4, 64), dtype=torch.float32)
            fixed_chars[:, :, :, 0] = 1.0
            chars_width = min(ref_atom_name_chars.shape[2], 4)
            vocab_width = min(ref_atom_name_chars.shape[3], 64)
            fixed_chars[:, :, :chars_width, :vocab_width] = ref_atom_name_chars[:, :l_atoms, :chars_width, :vocab_width]
            ref_atom_name_chars = fixed_chars
            changed = True
        feats["ref_atom_name_chars"] = ref_atom_name_chars
        return changed

    @staticmethod
    def _ensure_token_features(feats: dict[str, Any], l_tokens: int) -> bool:
        """Fill in the per-token features the confidence module expects."""
        changed = False
        num_tokens = int(getattr(boltz_const, "num_tokens", 33))

        if "res_type" not in feats:
            res_type = torch.zeros((1, l_tokens, num_tokens), dtype=torch.float32)
            res_type[:, :, 0] = 1.0
            feats["res_type"] = res_type
            changed = True

        if "profile" not in feats:
            profile = torch.zeros((1, l_tokens, num_tokens), dtype=torch.float32)
            msa_profile = feats.get("msa_profile")
            if torch.is_tensor(msa_profile):
                width = min(int(msa_profile.shape[-1]), num_tokens)
                profile[:, :, :width] = msa_profile[:, :, :width].to(dtype=torch.float32)
            feats["profile"] = profile
            changed = True

        if "method_feature" not in feats:
            feats["method_feature"] = torch.zeros((1, l_tokens), dtype=torch.int64)
            changed = True

        if "modified" not in feats:
            feats["modified"] = torch.zeros((1, l_tokens), dtype=torch.int64)
            changed = True

        if "cyclic_period" not in feats:
            feats["cyclic_period"] = torch.zeros((1, l_tokens), dtype=torch.float32)
            changed = True

        return changed

    @staticmethod
    def _ensure_contact_features(feats: dict[str, Any], l_tokens: int) -> bool:
        """Fill in the token-pair contact conditioning features."""
        changed = False

        if "contact_conditioning" not in feats:
            contact_conditioning_size = len(
                getattr(boltz_const, "contact_conditioning_info", {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
            )
            contact_conditioning = torch.zeros(
                (1, l_tokens, l_tokens, contact_conditioning_size),
                dtype=torch.float32,
            )
            contact_conditioning[:, :, :, 0] = 1.0
            feats["contact_conditioning"] = contact_conditioning
            changed = True

        if "contact_threshold" not in feats:
            feats["contact_threshold"] = torch.full((1, l_tokens, l_tokens), 8.0, dtype=torch.float32)
            changed = True

        return changed

    def _build_atom_to_token(
        self,
        l_atoms: int,
        l_tokens: int,
        token_atom_indices: list[list[int]] | None,
    ) -> torch.Tensor:
        atom_to_token = torch.zeros((1, l_atoms, l_tokens), dtype=torch.float32)
        if token_atom_indices:
            for token_id, atom_ids in enumerate(token_atom_indices[:l_tokens]):
                for atom_id in atom_ids:
                    if 0 <= atom_id < l_atoms:
                        atom_to_token[0, atom_id, token_id] = 1.0
        else:
            for atom_id in range(l_atoms):
                token_id = min((atom_id * l_tokens) // max(l_atoms, 1), l_tokens - 1)
                atom_to_token[0, atom_id, token_id] = 1.0
        return atom_to_token

    def _featurize_fallback(self, pdb_path: Path, pdb_id: str) -> dict[str, Any]:
        """Fallback featurizer for tests and environments without Boltz internals."""
        try:
            from Bio.PDB import PDBParser
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("biopython is required for fallback featurization") from exc

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(pdb_path))

        parsed = self._parse_fallback_structure(structure)
        if not parsed.atom_coords or not parsed.token_atom_indices:
            raise ValueError(f"No atoms/tokens parsed from {pdb_path}")

        feats = self._build_fallback_features(parsed=parsed, pdb_id=pdb_id)
        feats, _ = self._ensure_model_features(feats, token_atom_indices=parsed.token_atom_indices)
        return feats

    def _parse_fallback_structure(self, structure: Any) -> _ParsedFallbackStructure:
        """Collect atoms and residue tokens from the first model, honouring the size caps."""
        parsed = _ParsedFallbackStructure()
        chain_to_idx: dict[str, int] = {}

        first_model = next(iter(structure), None)
        chains = list(first_model) if first_model is not None else []
        for chain in chains:
            chain_idx = chain_to_idx.setdefault(chain.id, len(chain_to_idx))
            self._collect_chain_tokens(chain=chain, chain_idx=chain_idx, parsed=parsed)
            if len(parsed.atom_coords) >= self.max_atoms:
                break

        return parsed

    def _collect_chain_tokens(self, chain: Any, chain_idx: int, parsed: _ParsedFallbackStructure) -> None:
        """Append one token per residue of ``chain`` that contributed at least one atom."""
        for residue in chain:
            if len(parsed.token_atom_indices) >= self.max_tokens:
                break

            residue_atoms = self._collect_residue_atoms(residue=residue, parsed=parsed)
            if not residue_atoms:
                continue

            parsed.token_atom_indices.append(residue_atoms)
            parsed.residue_index.append(int(residue.id[1]))
            parsed.chain_index.append(chain_idx)

    def _collect_residue_atoms(self, residue: Any, parsed: _ParsedFallbackStructure) -> list[int]:
        """Append the atoms of ``residue`` and return their indices in the flat atom list."""
        residue_atoms: list[int] = []
        for atom in residue.get_atoms():
            if len(parsed.atom_coords) >= self.max_atoms:
                break
            residue_atoms.append(len(parsed.atom_coords))
            parsed.atom_coords.append([float(x) for x in atom.coord])
            occupancy = atom.occupancy if atom.occupancy is not None else 1.0
            parsed.atom_resolved.append(1.0 if occupancy > 0 else 0.0)
        return residue_atoms

    def _build_fallback_features(self, parsed: _ParsedFallbackStructure, pdb_id: str) -> dict[str, Any]:
        """Turn parsed atom/token records into the tensors listed in REQUIRED_KEYS."""
        l_atoms = len(parsed.atom_coords)
        l_tokens = len(parsed.residue_index)

        token_index = torch.arange(l_tokens, dtype=torch.int64).view(1, l_tokens)
        residue_index = torch.tensor(parsed.residue_index, dtype=torch.int64).view(1, l_tokens)
        chain_index = torch.tensor(parsed.chain_index, dtype=torch.int64).view(1, l_tokens)

        msa = token_index.clone().unsqueeze(1)
        msa_profile = torch.zeros((1, l_tokens, 22), dtype=torch.float32)
        msa_profile[:, :, :20] = 1.0 / 20.0

        frames_idx, frame_resolved_mask = self._build_fallback_frames(
            token_atom_indices=parsed.token_atom_indices,
            l_atoms=l_atoms,
            l_tokens=l_tokens,
        )

        return {
            "coords": torch.tensor(parsed.atom_coords, dtype=torch.float32).view(1, 1, l_atoms, 3),
            "atom_pad_mask": torch.ones((1, l_atoms), dtype=torch.float32),
            "atom_resolved_mask": torch.tensor(parsed.atom_resolved, dtype=torch.float32).view(1, l_atoms),
            "token_pad_mask": torch.ones((1, l_tokens), dtype=torch.float32),
            "token_index": token_index,
            "residue_index": residue_index,
            "chain_index": chain_index,
            "asym_id": chain_index.clone(),
            "entity_id": chain_index.clone(),
            "sym_id": torch.zeros((1, l_tokens), dtype=torch.int64),
            "mol_type": torch.zeros((1, l_tokens), dtype=torch.int64),
            "token_bonds": self._build_fallback_token_bonds(
                chain_index=chain_index,
                residue_index=residue_index,
                l_tokens=l_tokens,
            ),
            "msa": msa,
            "msa_mask": torch.ones_like(msa, dtype=torch.float32),
            "msa_profile": msa_profile,
            "deletion_mean": torch.zeros((1, l_tokens), dtype=torch.float32),
            "frames_idx": frames_idx,
            "frame_resolved_mask": frame_resolved_mask,
            "pdb_id": [pdb_id],
            "idx_dataset": torch.tensor([0], dtype=torch.int64),
        }

    @staticmethod
    def _build_fallback_token_bonds(
        chain_index: torch.Tensor,
        residue_index: torch.Tensor,
        l_tokens: int,
    ) -> torch.Tensor:
        """Bond consecutive residues that share a chain and are sequence neighbours."""
        token_bonds = torch.zeros((1, l_tokens, l_tokens, 1), dtype=torch.float32)
        for i in range(l_tokens - 1):
            same_chain = chain_index[0, i] == chain_index[0, i + 1]
            adjacent = residue_index[0, i + 1] - residue_index[0, i] == 1
            if bool(same_chain and adjacent):
                token_bonds[0, i, i + 1, 0] = 1.0
                token_bonds[0, i + 1, i, 0] = 1.0
        return token_bonds

    @staticmethod
    def _build_fallback_frames(
        token_atom_indices: list[list[int]],
        l_atoms: int,
        l_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a frame per token from its first three in-range atoms."""
        frames_idx = torch.zeros((1, 1, l_tokens, 3), dtype=torch.int64)
        frame_resolved_mask = torch.zeros((1, 1, l_tokens), dtype=torch.float32)
        for i, atom_ids in enumerate(token_atom_indices):
            valid = [atom_id for atom_id in atom_ids if atom_id < l_atoms]
            if len(valid) >= 3:
                frames_idx[0, 0, i] = torch.tensor(valid[:3], dtype=torch.int64)
                frame_resolved_mask[0, 0, i] = 1.0
        return frames_idx, frame_resolved_mask
