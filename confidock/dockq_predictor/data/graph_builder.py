"""Graph construction for antibody-antigen complexes."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser
from torch_geometric.data import Data

# 3-letter -> 1-letter residue codes, shared with the CDR/chain-typing path so the
# two can never disagree about what an unknown residue maps to ("X").
from .chain_mapping_cif import _THREE_TO_ONE


def fix_esmfold2_cif(cif_path: str) -> str:
    """
    Fix ESMFold2 CIF files missing occupancy field.

    Adds _atom_site.occupancy header and value 1.0 for all atoms.
    """
    import tempfile

    with open(cif_path, "r") as f:
        lines = f.readlines()

    # Check if already has occupancy
    has_occupancy = any("_atom_site.occupancy" in line for line in lines)
    if has_occupancy:
        # No fix needed - just return original
        return cif_path

    # Find position to insert occupancy header (before _atom_site.id which is usually last)
    new_lines = []

    for line in lines:
        # Insert occupancy header right before _atom_site.id
        if line.startswith("_atom_site.id"):
            new_lines.append("_atom_site.occupancy \n")
            new_lines.append(line)
        elif line.startswith("ATOM") or line.startswith("HETATM"):
            # Data line - insert 1.0 as second-to-last field (before the id field)
            parts = line.rstrip().split()
            if len(parts) >= 2:
                # Insert 1.0 before the last field (id)
                modified = " ".join(parts[:-1]) + " 1.0 " + parts[-1] + "\n"
                new_lines.append(modified)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        f.writelines(new_lines)
        return f.name


def fix_boltz2_pdb(pdb_path: str) -> tuple:
    """
    Fix Boltz2 PDB files that have multi-character chain IDs.

    Boltz2 outputs chain IDs like 'Hc', 'Lc' which violate PDB format.
    This function creates a temporary fixed PDB with single-char chain IDs.

    A 2-char chain ID pushes the rest of the line one column right, so for THOSE
    lines the substitution must consume both columns (21:23) to pull the coordinate
    fields back onto their spec positions 31-38 / 39-46 / 47-54. But boltz2 mixes
    widths within one file -- 13.6% of the .pdb files in the full manifest carry both
    a 2-char antibody chain ('Hc'/'Lc'/'Kx') and a 1-char antigen chain ('A'/'X') --
    and a 1-char line is ALREADY correctly aligned. Consuming 2 columns there deletes
    a real column and shifts the coordinates LEFT by one, which is silently harmless
    only while every coordinate is narrower than its 8-column field. It breaks as soon
    as one needs all 8 (a value <= -100 or >= 1000): 'x=14.275, y=-101.187' becomes the
    unparseable token '14.275-'. Hence the per-line width below.

    Measured over all 49,369 manifest .pdb files: exactly 1 (3fku_HcLcA_seed42_sample1,
    y=-101.187) actually failed, which is why this stayed hidden -- it surfaced only as
    a repeated "Invalid or missing coordinate(s) at line 1760" that
    ``FilteredDockQDataset`` swallowed by resampling. Verified after the fix: that file
    parses to all 4,324 atoms, and coordinates on the previously-working mixed files are
    byte-identical to before (the 1-char branch is what changed, and those lines were
    the broken ones).

    Returns (fixed_pdb_path, chain_id_map) where chain_id_map is {old_id: new_id}.
    """
    import tempfile

    # Map multi-char chain IDs to single chars
    chain_map = {}
    next_chain_id = ord("A")

    fixed_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # Extract chain ID from column 21-22
                orig_chain = line[21:23].strip()

                if orig_chain not in chain_map:
                    chain_map[orig_chain] = chr(next_chain_id)
                    next_chain_id += 1

                # Replace exactly as many columns as the original ID occupied, so a
                # 2-char ID is de-shifted and a 1-char ID keeps its alignment.
                width = 2 if len(orig_chain) == 2 else 1
                fixed_line = line[:21] + chain_map[orig_chain] + line[21 + width :]
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.writelines(fixed_lines)
        return f.name, chain_map


def _chain_sequence(chain) -> str:
    """Read a chain's standard residues as a 1-letter sequence (unknowns -> "X")."""
    return "".join(_THREE_TO_ONE.get(residue.resname, "X") for residue in chain if residue.id[0] == " ")


def _anarci_chain_type(chain_id: str, seq: str) -> str:
    """Type one chain by running ANARCI over its sequence.

    Returns "VH", "VL", or "antigen" (also when ANARCI runs but numbers nothing).
    Propagates ``TimeoutExpired`` / ``FileNotFoundError`` so the caller can fall
    back to the length heuristic when ANARCI is missing or hangs.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">chain_{chain_id}\n{seq}\n")
        fasta_path = f.name

    result = subprocess.run(
        ["ANARCI", "-i", fasta_path, "--scheme", "chothia"], capture_output=True, text=True, timeout=30
    )

    os.unlink(fasta_path)

    # Parse ANARCI output
    if result.returncode != 0 or not result.stdout:
        return "antigen"

    output = result.stdout
    if "VH" in output or "H|" in output:
        return "VH"
    if "VL" in output or "L|" in output or "K|" in output:
        return "VL"
    return "antigen"


def _length_heuristic_chain_type(seq: str) -> str:
    """Guess a chain's type from its length alone, when ANARCI is unavailable.

    Heavy chains are typically 110-130 residues and light chains 100-120, so
    anything in 100-140 is provisionally "antibody" -- a label
    ``_resolve_antibody_labels`` later splits into VH/VL.
    """
    return "antibody" if 100 <= len(seq) <= 140 else "antigen"


def _resolve_antibody_labels(chain_types: Dict[str, str], pdb_path: str) -> None:
    """Split a pair of provisional "antibody" labels into VH/VL, in place.

    Only fires for exactly two such chains, where the longer is assumed heavy and
    the shorter light. Any other count is left alone.
    """
    antibody_chains = [cid for cid, ctype in chain_types.items() if ctype == "antibody"]
    if len(antibody_chains) != 2:
        return

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    lengths = {}
    for model in structure:
        for chain in model:
            if chain.id in antibody_chains:
                lengths[chain.id] = len([r for r in chain if r.id[0] == " "])

    sorted_chains = sorted(antibody_chains, key=lambda x: lengths[x], reverse=True)
    chain_types[sorted_chains[0]] = "VH"
    chain_types[sorted_chains[1]] = "VL"


def identify_antibody_chains(pdb_path: str) -> Dict[str, str]:
    """
    Identify antibody chains (VH, VL) and antigen chains using ANARCI.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file

    Returns
    -------
    Dict[str, str]
        Dictionary mapping chain_id to chain_type ('VH', 'VL', or 'antigen')
    """
    chain_types: Dict[str, str] = {}

    # Parse PDB to get sequences
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    for model in structure:
        for chain in model:
            seq = _chain_sequence(chain)
            try:
                chain_types[chain.id] = _anarci_chain_type(chain.id, seq)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # ANARCI not available or timed out - fall back to the heuristic
                chain_types[chain.id] = _length_heuristic_chain_type(seq)

    # Post-process: if we have 'antibody' labels, assign VH/VL
    _resolve_antibody_labels(chain_types, pdb_path)

    return chain_types


def _residue_geometry(residue) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Backbone coords and heavy-atom coords for one Bio.PDB residue.

    Returns ``(backbone_coords, heavy_atom_coords)`` where backbone_coords maps
    the present N/CA/C/O atom names to coords and heavy_atom_coords is an
    ``[n_heavy, 3]`` array of all non-hydrogen atom coords. These feed the two
    kinds of geometry that need more than the CA position: the local backbone
    frames (relative_rotation, translation_*_frame) and min_heavy_atom_dist.
    """
    bb = {atom_name: residue[atom_name].get_coord() for atom_name in ("N", "CA", "C", "O") if atom_name in residue}
    heavy = np.array([a.get_coord() for a in residue if a.element != "H"], dtype=np.float32)
    return bb, heavy


def _knn_edges_from_distances(dmat: torch.Tensor, knn_k: int) -> torch.Tensor:
    """k-NN ``edge_index`` from a precomputed pairwise distance matrix.

    Pure-torch replacement for ``torch_cluster.knn_graph``, which needs a
    compiled extension built against the exact torch build. The distance matrix
    is computed anyway for the cross-chain and interface masks, so a ``topk``
    over it is cheaper than a second neighbour search.

    Parameters
    ----------
    dmat : torch.Tensor
        ``[N, N]`` pairwise distances.
    knn_k : int
        Neighbours per node, capped at ``N - 1`` (self-loops excluded).

    Returns
    -------
    torch.Tensor
        ``[2, N * k]`` edge index oriented neighbour -> query, i.e. the
        ``flow="source_to_target"`` convention PyG message passing expects.
    """
    n = dmat.shape[0]
    k = min(knn_k, n - 1)
    if k <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Mask the diagonal so a node is never its own neighbour.
    masked = dmat.clone()
    masked.fill_diagonal_(float("inf"))
    neighbours = masked.topk(k, dim=1, largest=False).indices  # [N, k]
    target = torch.arange(n, dtype=torch.long).repeat_interleave(k)
    return torch.stack([neighbours.reshape(-1), target], dim=0)


def _load_cdr_labels(pdb_path: Optional[str], structure, chain_types: Dict[str, str]) -> Dict:
    """Read real CDR labels from the SAbDab JSON (h_cdr/l_cdr, by residue order).

    Empty dict when there is no path, no JSON, or no antibody chains, so labels
    default to 0 rather than failing the build.
    """
    if pdb_path is None:
        return {}

    from .chain_mapping_cif import compute_cdr_labels_from_json

    try:
        return compute_cdr_labels_from_json(pdb_path, structure, chain_types)
    except Exception as e:
        print(f"Warning: CDR labeling failed for {pdb_path}: {e}")
        return {}


def _iter_standard_residues(structure):
    """Yield ``(chain_id, residue)`` for each standard residue that has a CA atom."""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " " and "CA" in residue:
                    yield chain.id, residue


def _extract_residue_nodes(
    structure,
    chain_types: Dict[str, str],
    chain_id_map: Dict[str, int],
    cdr_labels: Dict,
) -> Tuple[list, list, list, list]:
    """Walk the structure into the per-node lists the graph is assembled from.

    Returns ``(residues, positions, plddts, chain_ids)``, one entry per node,
    where a node is a standard residue with a CA atom (the graph's CA-only rule).
    """
    residues: list = []
    positions: list = []
    plddts: list = []
    chain_ids: list = []

    for chain_id, residue in _iter_standard_residues(structure):
        # Get CA position and pLDDT
        ca_atom = residue["CA"]
        res_num = residue.id[1]
        insertion_code = residue.id[2] if len(residue.id) > 2 else ""
        positions.append(ca_atom.get_coord())
        plddts.append(ca_atom.get_bfactor())
        # Backbone + heavy-atom coords for the geometric edge features (local
        # frames, min-heavy-atom distance). Kept on the residue dict, which the
        # feature builders already receive and which is dropped before the graph
        # is cached -- so this adds no cache/signature cost.
        bb, heavy = _residue_geometry(residue)
        residues.append(
            {
                "chain_id": chain_id,
                "chain_type": chain_types.get(chain_id, "antigen"),
                "residue_number": res_num,
                "insertion_code": insertion_code,
                "residue_name": residue.resname,
                "cdr_label": cdr_labels.get((chain_id, res_num, insertion_code), 0),
                "backbone_coords": bb,
                "heavy_atom_coords": heavy,
            }
        )
        chain_ids.append(chain_id_map[chain_id])

    return residues, positions, plddts, chain_ids


def _cross_chain_edges(cross_chain: torch.Tensor, dmat: torch.Tensor, n_residues: int) -> torch.Tensor:
    """Cross-chain residue pairs within 12A as a bidirectional ``edge_index``.

    Built from the upper triangle and mirrored, so each qualifying pair yields
    both directions. Empty ``[2, 0]`` when no pair qualifies.
    """
    within12 = cross_chain & (dmat <= 12.0)
    ii, jj = torch.triu_indices(n_residues, n_residues, offset=1).unbind(0)
    keep = within12[ii, jj]
    src, dst = ii[keep], jj[keep]
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _residue_flags(residues: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-node ``(is_antibody, is_heavy, cdr_label)`` tensors."""
    n_residues = len(residues)
    is_antibody = torch.zeros(n_residues, dtype=torch.bool)
    is_heavy = torch.zeros(n_residues, dtype=torch.bool)
    cdr_label_tensor = torch.zeros(n_residues, dtype=torch.long)
    for i, res in enumerate(residues):
        chain_type = res["chain_type"]
        is_antibody[i] = chain_type in ["VH", "VL"]
        is_heavy[i] = chain_type == "VH"
        cdr_label_tensor[i] = res["cdr_label"]
    return is_antibody, is_heavy, cdr_label_tensor


def _build_residue_graph(
    structure,
    chain_types: Dict[str, str],
    chain_id_map: Dict[str, int],
    interface_cutoff: float,
    knn_k: int,
    pdb_path: Optional[str] = None,
) -> Tuple[Data, Dict]:
    """Build residue-level graph (CA atoms only)."""
    cdr_labels = _load_cdr_labels(pdb_path, structure, chain_types)

    # Extract residues and features
    residues, positions, plddts, chain_ids = _extract_residue_nodes(structure, chain_types, chain_id_map, cdr_labels)
    n_residues = len(residues)

    # Convert to tensors
    pos = torch.tensor(np.array(positions), dtype=torch.float32)
    chain_id_tensor = torch.tensor(chain_ids, dtype=torch.long)

    # Pairwise residue distances and a cross-chain mask, computed once as
    # vectorized tensor ops (replaces the former O(n^2) Python double-loops).
    dmat = torch.cdist(pos, pos)  # [N, N]
    cross_chain = chain_id_tensor[:, None] != chain_id_tensor[None, :]  # [N, N]

    # Build the k-NN graph from the distance matrix we already have, so the
    # builder needs no compiled torch_cluster/pyg-lib extension. Edges point
    # neighbour -> query, matching torch_cluster.knn_graph(flow="source_to_target").
    edge_index_knn = _knn_edges_from_distances(dmat, knn_k)

    # Add cross-chain edges within 12A (upper triangle -> both directions).
    cross_chain_edge_index = _cross_chain_edges(cross_chain, dmat, n_residues)
    edge_index = torch.cat([edge_index_knn, cross_chain_edge_index], dim=1)

    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    # Interface residues: any cross-chain partner within the cutoff.
    is_interface = (cross_chain & (dmat <= interface_cutoff)).any(dim=1)

    is_antibody, is_heavy, cdr_label_tensor = _residue_flags(residues)

    # The N/CA/C backbone frames are NOT stored on the graph. They are computed
    # inside compute_edge_features and consumed there, landing in edge_attr as
    # pre-flattened scalars (relative_rotation, translation_*_frame). Only an
    # equivariant stack would need them kept as [N, 3, 3] vectors that rotate with
    # the structure, and this model is not equivariant.

    # Create Data object
    data = Data(
        pos=pos,
        edge_index=edge_index,
        chain_id=chain_id_tensor,
        is_interface=is_interface,
        is_antibody=is_antibody,
        is_heavy=is_heavy,
        cdr_label=cdr_label_tensor,
        num_nodes=n_residues,
    )

    data.residues = residues
    data.plddt = torch.tensor(plddts, dtype=torch.float32)

    return data, {"residues": residues, "chain_types": chain_types}


def _parse_cif_structure(cif_path: str) -> Tuple[object, Optional[str]]:
    """Parse a CIF, repairing a missing occupancy column if that is why it failed.

    Returns ``(structure, temp_path)``, where temp_path is the repaired copy the
    caller must clean up, or None when no repair was needed.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        return parser.get_structure("complex", cif_path), None
    except KeyError as e:
        # ESMFold2 CIF files may be missing occupancy
        if "occupancy" not in str(e):
            raise ValueError(f"Failed to parse CIF: {e}") from e
        temp_path = fix_esmfold2_cif(cif_path)
        return parser.get_structure("complex", temp_path), temp_path


def _pdb_needs_chain_fix(pdb_path: str) -> bool:
    """Check whether any ATOM/HETATM line carries a multi-character chain ID."""
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and len(line[21:23].strip()) > 1:
                    return True
    except Exception:
        pass
    return False


def _remap_chain_types(chain_types: Dict[str, str], chain_id_map: Dict[str, str]) -> Dict[str, str]:
    """Re-key chain types onto the renamed chains of a repaired PDB."""
    return {new_id: chain_types[old_id] for old_id, new_id in chain_id_map.items() if old_id in chain_types}


def _parse_pdb_structure(
    pdb_path: str, chain_types: Optional[Dict[str, str]]
) -> Tuple[object, Optional[str], Optional[Dict[str, str]]]:
    """Parse a PDB, first repairing multi-char boltz2 chain IDs if present.

    Returns ``(structure, temp_path, chain_types)``; chain_types is re-keyed when
    the repair renamed the chains.
    """
    # Check for multi-char chain IDs BEFORE parsing, since PERMISSIVE mode will
    # mangle the data.
    parser = PDBParser(QUIET=True, PERMISSIVE=True)

    if _pdb_needs_chain_fix(pdb_path):
        temp_path, chain_id_map = fix_boltz2_pdb(pdb_path)
        structure = parser.get_structure("complex", temp_path)
        if chain_types is not None:
            chain_types = _remap_chain_types(chain_types, chain_id_map)
        return structure, temp_path, chain_types

    try:
        return parser.get_structure("complex", pdb_path), None, chain_types
    except (ValueError, KeyError) as e:
        raise ValueError(f"Failed to parse structure: {e}") from e


def _parse_structure(
    pdb_path: str, chain_types: Optional[Dict[str, str]]
) -> Tuple[object, Optional[str], Optional[Dict[str, str]]]:
    """Parse a PDB or CIF file, choosing the parser by extension."""
    if Path(pdb_path).suffix.lower() in [".cif", ".mmcif"]:
        structure, temp_path = _parse_cif_structure(pdb_path)
        return structure, temp_path, chain_types
    return _parse_pdb_structure(pdb_path, chain_types)


def _chain_lengths(structure) -> Dict[str, int]:
    """Count the standard residues of each chain present in the structure."""
    lengths: Dict[str, int] = {}
    for model in structure:
        for chain in model:
            lengths[chain.id] = len([r for r in chain if r.id[0] == " "])
    return lengths


def _chain_id_map(structure) -> Dict[str, int]:
    """Assign each chain a 0..n-1 index in structure order, for consistent ids."""
    chain_id_map: Dict[str, int] = {}
    for model in structure:
        for chain in model:
            if chain.id not in chain_id_map:
                chain_id_map[chain.id] = len(chain_id_map)
    return chain_id_map


def _resolve_chain_types(
    chain_types: Optional[Dict[str, str]],
    chain_lengths: Dict[str, int],
    pdb_path: str,
    analysis_path: str,
) -> Dict[str, str]:
    """Settle chain typing: provided annotation, else SAbDab JSON, else ANARCI.

    A provided annotation is only trusted when its chain IDs actually occur in
    the structure -- e.g. SAbDab symbolic chains against a CIF's numeric ones do
    not match, and would otherwise silently type every chain as antigen.
    """
    if chain_types is not None and set(chain_types).issubset(chain_lengths):
        return chain_types

    # Try sequence-based matching from JSON first (for all files)
    from .chain_mapping_cif import identify_chains_from_json

    chain_types = identify_chains_from_json(pdb_path, chain_lengths)

    if not chain_types:
        # Fall back to ANARCI if JSON not found
        print(f"Warning: Could not identify chains from JSON for {Path(pdb_path).name}, using ANARCI...")
        chain_types = identify_antibody_chains(analysis_path)

    return chain_types


def _remove_temp_file(path: Optional[str]) -> None:
    """Delete a repaired temporary structure file, ignoring failures."""
    if path is None:
        return
    try:
        os.unlink(path)
    except Exception:
        pass


def build_graph(
    pdb_path: str,
    interface_cutoff: float = 8.0,
    knn_k: int = 16,
    chain_types: Optional[Dict[str, str]] = None,
) -> Data:
    """
    Build a residue-level graph from a PDB or CIF file.

    One node per residue, positioned at its CA. Edges are the union of a k-NN
    graph over CA positions and every cross-chain pair within 12A -- the k-NN part
    keeps the graph connected along each chain, the cross-chain part guarantees the
    interface is densely covered regardless of how k falls out.

    Node/edge FEATURES are not attached here; the caller applies
    ``compute_node_features`` / ``compute_edge_features`` (see
    ``DockQGraphDataset``). This function produces topology, positions and the
    per-residue metadata those need.

    Parameters
    ----------
    pdb_path : str
        Path to PDB or CIF file
    interface_cutoff : float
        Distance cutoff for flagging interface residues (Angstroms)
    knn_k : int
        Number of nearest neighbors for the k-NN graph
    chain_types : Dict[str, str], optional
        Pre-computed chain type annotations. When absent, or when the provided
        chain IDs don't match the structure's, they are recovered from the SAbDab
        JSON and finally from ANARCI.

    Returns
    -------
    torch_geometric.data.Data
        Graph topology, positions, pLDDT, CDR labels and chain annotations
    """
    # Parse PDB or CIF file, repairing the known malformations of either
    structure, temp_pdb, chain_types = _parse_structure(pdb_path, chain_types)

    # Use temp_pdb path if we fixed the file
    analysis_path = temp_pdb if temp_pdb is not None else pdb_path

    # Identify chain types
    chain_lengths = _chain_lengths(structure)
    chain_types = _resolve_chain_types(chain_types, chain_lengths, pdb_path, analysis_path)

    data, _metadata = _build_residue_graph(
        structure, chain_types, _chain_id_map(structure), interface_cutoff, knn_k, pdb_path=pdb_path
    )

    # Add chain_types to data
    data.chain_types = chain_types

    # Clean up temp file if created
    _remove_temp_file(temp_pdb)

    return data
