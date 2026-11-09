"""DockQ evaluation and structure preprocessing utilities.

This module provides model-agnostic functions for running DockQ evaluation
and preprocessing protein structure files. These utilities work with predictions
from any source (Boltz-2, Protenix, ESMFold2, etc.).

Key functions:
- `run_dockq()`: Computes per-interface DockQ metrics
- `run_complex_level_dockq()`: Treats multiple binder chains as a single unit
- `run_dockq_with_complex_score()`: Combines both per-interface and complex-level scoring
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _is_hydrogen_atom(atom_name: str, element: str) -> bool:
    """Check if an atom is a hydrogen based on name or element.

    Parameters
    ----------
    atom_name : str
        Atom name from columns 13-16 of PDB record
    element : str
        Element symbol from columns 77-78 of PDB record

    Returns
    -------
    bool
        True if the atom is a hydrogen
    """
    # Check element field
    if element == "H":
        return True

    # Check atom name for old-style hydrogen naming
    if atom_name.startswith("H"):
        return True

    # Check for digit followed by H (e.g. 1HG2, 2HG2)
    if len(atom_name) >= 2 and atom_name[0].isdigit() and atom_name[1] == "H":
        return True

    return False


def _strip_hydrogens(pdb_text: str) -> str:
    """Return PDB text with hydrogen ATOM/HETATM records removed.

    Old-style PDB hydrogen naming (e.g. ``1HG2``, ``2HG2``) can cause parse
    failures in DockQ's parser. DockQ does not use H atoms for any of its
    metrics (Fnat, iRMSD, LRMSD all use heavy atoms / Cα only), so stripping
    them is safe.

    Parameters
    ----------
    pdb_text : str
        Raw PDB file content as text

    Returns
    -------
    str
        PDB text with all hydrogen atoms removed
    """
    lines: list[str] = []
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"}:
            atom_name = line[12:16].strip() if len(line) >= 16 else ""
            element = line[76:78].strip() if len(line) >= 78 else ""
            if _is_hydrogen_atom(atom_name, element):
                continue
        lines.append(line)
    return "\n".join(lines)


def _identify_chain_ids(lines: list[str]) -> tuple[set[str], set[str]]:
    """Identify single-char and multi-char chain IDs in PDB lines.

    Parameters
    ----------
    lines : list of str
        PDB file lines

    Returns
    -------
    tuple of (set, set)
        (single_ids, multi_ids) where single_ids are 1-char chain IDs
        and multi_ids are 2-char chain IDs
    """
    single_ids: set[str] = set()
    multi_ids: set[str] = set()
    for line in lines:
        if len(line) < 22 or line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        c1, c2 = line[21], line[22]
        if c2.isalpha():
            multi_ids.add(c1 + c2)
        else:
            single_ids.add(c1)
    return single_ids, multi_ids


def _create_chain_id_remapping(single_ids: set[str], multi_ids: set[str]) -> dict[str, str]:
    """Create remapping from multi-char chain IDs to single-char IDs.

    Parameters
    ----------
    single_ids : set of str
        Existing single-char chain IDs
    multi_ids : set of str
        Multi-char chain IDs to remap

    Returns
    -------
    dict
        Mapping from chain ID to single-char replacement
    """
    used: set[str] = set(single_ids)
    candidates = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    remap: dict[str, str] = {c: c for c in single_ids}
    for cid in sorted(multi_ids):
        preferred = cid[0].upper()
        replacement = preferred if preferred not in used else next(c for c in candidates if c not in used)
        remap[cid] = replacement
        used.add(replacement)
    return remap


def _rewrite_pdb_lines_with_remap(lines: list[str], remap: dict[str, str]) -> list[str]:
    """Rewrite PDB lines with remapped chain IDs.

    Parameters
    ----------
    lines : list of str
        Original PDB lines
    remap : dict
        Chain ID remapping

    Returns
    -------
    list of str
        Rewritten PDB lines
    """
    out_lines: list[str] = []
    for line in lines:
        if len(line) >= 22 and line[:6].strip() in {"ATOM", "HETATM"}:
            c1, c2 = line[21], line[22]
            if c2.isalpha():
                new_chain = remap.get(c1 + c2, c1)
                line = line[:21] + new_chain + line[23:]
        out_lines.append(line)
    return out_lines


def _normalise_pdb_chains(pdb_text: str) -> tuple[str, dict[str, str]]:
    """Remap non-standard 2-character chain IDs to single-character IDs.

    Some structure prediction tools (e.g., Boltz-2) output 2-character chain IDs
    (e.g. ``Hc``, ``Lc``) at columns 21-22 of ATOM/HETATM records, which violates
    the PDB standard (chain ID must be exactly 1 character) and causes DockQ's
    parser to fail.

    This function detects affected records, assigns each unique 2-char chain ID
    a distinct single-char replacement, and rewrites the lines to standard
    column alignment.

    Parameters
    ----------
    pdb_text : str
        Raw PDB file content as text

    Returns
    -------
    tuple[str, dict[str, str]]
        ``(normalised_text, remap)`` where *remap* maps every chain ID seen in
        the file to its (possibly unchanged) single-char replacement.

    Examples
    --------
    >>> text, remap = _normalise_pdb_chains(pdb_with_hc_lc_chains)
    >>> remap
    {'Hc': 'H', 'Lc': 'L', 'A': 'A'}
    """
    lines = pdb_text.splitlines()
    single_ids, multi_ids = _identify_chain_ids(lines)

    # If no multi-char chains, return as-is
    if not multi_ids:
        return pdb_text, {c: c for c in single_ids}

    # Create remapping and rewrite PDB
    remap = _create_chain_id_remapping(single_ids, multi_ids)
    out_lines = _rewrite_pdb_lines_with_remap(lines, remap)

    return "\n".join(out_lines), remap


def _add_occupancy_to_atom_line(line: str) -> str:
    """Add occupancy value to CIF atom line if it has enough fields."""
    parts = line.split()
    if len(parts) >= 15:
        parts.insert(14, "1.0")
        return " ".join(parts)
    return line


def _is_atom_line(line: str) -> bool:
    """Check if line is an ATOM or HETATM line."""
    return line.startswith("ATOM") or line.startswith("HETATM")


def _process_cif_line(line: str, header_done: bool) -> str:
    """Process a CIF line, adding occupancy to atom lines if needed."""
    if header_done and _is_atom_line(line):
        return _add_occupancy_to_atom_line(line)
    return line


def _add_occupancy_field_to_cif(cif_text: str) -> str:
    """Add missing occupancy field to CIF file.

    Parameters
    ----------
    cif_text : str
        CIF file content

    Returns
    -------
    str
        CIF content with occupancy field added
    """
    lines = cif_text.split("\n")
    new_lines = []
    in_atom_site_header = False
    header_done = False

    for line in lines:
        if line.startswith("_atom_site."):
            in_atom_site_header = True
            new_lines.append(line)
            if "_atom_site.B_iso_or_equiv" in line:
                new_lines.append("_atom_site.occupancy")
        elif in_atom_site_header:
            # Exiting header section
            header_done = True
            in_atom_site_header = False
            new_lines.append(_process_cif_line(line, header_done=True))
        else:
            new_lines.append(_process_cif_line(line, header_done))

    return "\n".join(new_lines)


def _parse_cif_with_occupancy_fix(cif_path: Path, cif_text: str):
    """Parse CIF file, adding occupancy field if missing.

    Parameters
    ----------
    cif_path : Path
        Original CIF file path
    cif_text : str
        CIF file content

    Returns
    -------
    Bio.PDB.Structure
        Parsed structure
    """
    from Bio.PDB import MMCIFParser

    if "_atom_site.occupancy" not in cif_text:
        cif_text = _add_occupancy_field_to_cif(cif_text)
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as tmp:
            tmp.write(cif_text)
            tmp_path = Path(tmp.name)

        try:
            parser = MMCIFParser(QUIET=True)
            return parser.get_structure("model", str(tmp_path))
        finally:
            tmp_path.unlink()
    else:
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure("model", str(cif_path))


def _find_available_chain_id(used_ids: set[str], already_assigned: set[str]) -> str | None:
    """Find an available single-character chain ID."""
    candidates = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for candidate in candidates:
        if candidate not in used_ids and candidate not in already_assigned:
            return candidate
    return None


def _find_multichar_chain_remapping(structure) -> dict[str, str]:
    """Find and create remapping for multi-character chain IDs.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Parsed structure

    Returns
    -------
    dict
        Mapping from multi-char chain ID to single-char replacement
    """
    multi_char_chains = {}
    for model in structure:
        used_ids = {c.id for c in model if len(c.id) == 1}
        for chain in model:
            chain_id = chain.id
            if len(chain_id) > 1 and chain_id not in multi_char_chains:
                new_id = _find_available_chain_id(used_ids, set(multi_char_chains.values()))
                if new_id:
                    multi_char_chains[chain_id] = new_id
    return multi_char_chains


def _rename_chains_in_structure(structure, multi_char_chains: dict[str, str]):
    """Rename chains in structure according to remapping.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Structure to modify
    multi_char_chains : dict
        Chain ID remapping
    """
    for model in structure:
        for chain in list(model):
            if chain.id in multi_char_chains:
                new_id = multi_char_chains[chain.id]
                chain.id = new_id


def _cif_to_pdb_text(cif_path: Path) -> tuple[str, dict[str, str]]:
    """Convert a CIF structure file to PDB-format text via BioPython.

    Handles CIF files that are missing the `_atom_site.occupancy` field by
    adding it with a default value of 1.0 before parsing. Also handles
    multi-character chain IDs by remapping them to single characters.

    Parameters
    ----------
    cif_path : Path
        Path to CIF file

    Returns
    -------
    tuple of (str, dict)
        (PDB-format text, chain_remap) where chain_remap maps new single-char
        IDs back to original multi-char IDs (e.g., {'A': 'Hc', 'B': 'Lc'})
    """
    import io

    from Bio.PDB import PDBIO

    # Parse CIF file
    cif_text = cif_path.read_text()
    structure = _parse_cif_with_occupancy_fix(cif_path, cif_text)

    # Handle multi-character chain IDs
    multi_char_chains = _find_multichar_chain_remapping(structure)
    if multi_char_chains:
        _rename_chains_in_structure(structure, multi_char_chains)

    # Convert to PDB format
    io_obj = PDBIO()
    io_obj.set_structure(structure)
    buf = io.StringIO()
    io_obj.save(buf)

    # Return the chain remapping (new_id -> original_id) for tracking
    chain_remap_out = {v: k for k, v in multi_char_chains.items()}
    return buf.getvalue(), chain_remap_out


def run_dockq(
    model_path: Path,
    native_path: Path,
    chain_map: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], float]:
    """Run DockQ evaluation for one model/native pair.

    Computes per-interface DockQ metrics (DockQ score, iRMSD, LRMSD, fnat, etc.)
    for all interfaces in the complex.

    This is model-agnostic and works with predictions from any source (Boltz-2,
    Protenix, ESMFold2, etc.) as long as structure files are provided.

    Parameters
    ----------
    model_path : Path
        Predicted structure (.pdb or .cif). CIF files are converted to PDB
        automatically via BioPython.
    native_path : Path
        Reference crystal structure (.pdb).
    chain_map : dict
        ``{native_chain: model_chain}`` mapping. Example: ``{'A': 'H', 'B': 'L'}``

    Returns
    -------
    tuple of (dict, float)
        ``(interfaces, global_dockq)`` where:

        - *interfaces*: dict mapping native-chain pair keys (e.g. ``"AB"``) to
          metric dicts containing ``DockQ``, ``iRMSD``, ``LRMSD``, ``fnat``,
          ``fnonnat``, ``F1``, ``clashes``, ``model_chains``
        - *global_dockq*: average DockQ score across all interfaces

    Raises
    ------
    ImportError
        If the ``DockQ`` package is not installed (``pip install dockq``).

    Examples
    --------
    >>> from confidock.evaluation import run_dockq
    >>> interfaces, global_score = run_dockq(
    ...     Path("model.pdb"),
    ...     Path("native.pdb"),
    ...     {'A': 'H', 'B': 'L', 'C': 'T'}
    ... )
    >>> interfaces['AB']['DockQ']
    0.7234
    """
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError as exc:
        raise ImportError("DockQ is not installed. Install it with: pip install dockq") from exc

    # Preprocess structures: strip hydrogens, normalize chains, convert CIF to PDB
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        # Use fixed, distinct filenames. Deriving names from the input stems can
        # collide when the model and native share a stem (e.g. ESMFold2 no_msa
        # single-sample predictions named "{entry}.cif" against "{entry}.pdb"),
        # in which case the native write clobbers the model and DockQ silently
        # scores the native against itself.
        sanitised_model = tmp / "model.pdb"
        sanitised_native = tmp / "native.pdb"

        # Load and preprocess model
        if model_path.suffix == ".cif":
            model_text, cif_chain_remap = _cif_to_pdb_text(model_path)
        else:
            model_text = model_path.read_text()
            cif_chain_remap = {}

        model_text, normalize_chain_remap = _normalise_pdb_chains(model_text)

        # Combine CIF and normalization remappings
        # chain_remap maps final_id -> original_id
        # normalize_chain_remap is {original_id: normalized_id}
        # We need to invert it to get {normalized_id: original_id}
        chain_remap = {}
        for orig_id, norm_id in normalize_chain_remap.items():
            # Check if this came from CIF conversion first
            actual_orig = cif_chain_remap.get(orig_id, orig_id)
            chain_remap[norm_id] = actual_orig

        # Update chain_map: if model chain IDs changed during conversion,
        # map them to the new (final) IDs
        # Invert chain_remap to get original_id -> final_id
        inverted_remap = {orig: final for final, orig in chain_remap.items()}
        original_chain_map = dict(chain_map)  # Save for debugging
        chain_map = {nat: inverted_remap.get(mod, mod) for nat, mod in chain_map.items()}
        LOGGER.debug(f"Chain map update: {original_chain_map} -> {chain_map} (inverted_remap={inverted_remap})")

        # Write preprocessed files
        sanitised_model.write_text(_strip_hydrogens(model_text))
        sanitised_native.write_text(_strip_hydrogens(native_path.read_text()))

        # Load structures with DockQ
        model = load_PDB(str(sanitised_model))
        native = load_PDB(str(sanitised_native))

    # Filter chain_map to chains that actually exist in both structures
    native_chain_ids = {c.id for c in native.get_chains()}
    model_chain_ids = {c.id for c in model.get_chains()}
    LOGGER.debug(f"Native chains: {native_chain_ids}, Model chains: {model_chain_ids}")
    LOGGER.debug(f"Checking chain_map: {chain_map}")
    filtered_chain_map = {
        nat: mod for nat, mod in chain_map.items() if nat in native_chain_ids and mod in model_chain_ids
    }
    LOGGER.debug(f"Filtered chain_map: {filtered_chain_map}")

    if len(filtered_chain_map) < len(chain_map):
        dropped = {k: v for k, v in chain_map.items() if k not in filtered_chain_map}
        LOGGER.warning("Dropped chain_map entries (chain not found in structure): %s", dropped)

    # Run DockQ on all interfaces
    interfaces_raw, global_dockq = run_on_all_native_interfaces(model, native, chain_map=filtered_chain_map)

    # Format results
    interfaces: dict[str, dict[str, Any]] = {}
    for iface_key, metrics in interfaces_raw.items():
        iface_str = str(iface_key)

        # Validate interface key uses native chain IDs
        chain_ids = list(iface_str[:2]) if len(iface_str) >= 2 else []
        for chain_id in chain_ids:
            if chain_id not in native_chain_ids:
                LOGGER.warning(
                    "Interface key '%s' contains chain '%s' not found in native structure "
                    "(native chains: %s). This may indicate DockQ returned model chain IDs.",
                    iface_str,
                    chain_id,
                    list(native_chain_ids),
                )

        interfaces[iface_str] = {
            "DockQ": metrics.get("DockQ"),
            "iRMSD": metrics.get("iRMSD"),
            "LRMSD": metrics.get("LRMSD"),
            "fnat": metrics.get("fnat"),
            "fnonnat": metrics.get("fnonnat"),
            "F1": metrics.get("F1"),
            "clashes": metrics.get("clashes"),
            "model_chains": metrics.get("chain1", "") + metrics.get("chain2", ""),
        }

    return interfaces, float(global_dockq)


def _merge_chains_in_structure(structure, chains_to_merge: list[str], new_chain_id: str):
    """Merge multiple chains into a single chain within a Bio.PDB structure.

    Based on the recommended approach from DockQ GitHub issue #33.
    Updated version that correctly handles the sequence attribute.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        Structure object to modify in-place (from DockQ's load_PDB)
    chains_to_merge : list of str
        Chain IDs to merge (e.g., ["H", "L"])
    new_chain_id : str
        New chain ID for the merged chains

    Returns
    -------
    Bio.PDB.Structure
        Modified structure with merged chains
    """
    # DockQ's load_PDB returns a structure where the model is accessed directly
    # It's not a standard Bio.PDB structure with integer model IDs
    model = structure  # The structure itself acts as the model

    # Collect sequences from chains to merge
    seqs_to_merge = [model[chain].sequence for chain in chains_to_merge]

    # Add residues from all chains following the first one
    # BioPython requires unique residue IDs within a chain. When merging chains,
    # we need to renumber residues to avoid ID collisions (e.g., both chains have residue 1).
    # We use a continuous numbering scheme starting from a large offset to avoid conflicts.
    offset = 10000  # Large offset to avoid collision with existing residue numbers
    for chain_idx, chain_id in enumerate(chains_to_merge[1:], start=1):
        for res_idx, res in enumerate(model[chain_id]):
            # Renumber residue to avoid ID collision
            # Keep hetfield and insertion code, but use unique sequence number
            original_id = res.id
            new_seqnum = offset * chain_idx + res_idx
            res.id = (original_id[0], new_seqnum, original_id[2])
            model[chains_to_merge[0]].add(res)
        model.detach_child(chain_id)

    # Create merged chain name and sequence
    new_merged_chain_sequence = "".join(seqs_to_merge)

    # Rename the first chain with new merged ID
    model[chains_to_merge[0]].id = new_chain_id

    # Add corresponding merged sequence
    model[new_chain_id].sequence = new_merged_chain_sequence

    return structure


def _preprocess_structures_for_complex_dockq(
    model_path: Path,
    native_path: Path,
    chain_map: dict[str, str],
    tmp: Path,
) -> tuple[str, dict[str, str], dict[str, str], Path, Path]:
    """Preprocess model and native structures for complex-level DockQ.

    Returns
    -------
    tuple
        (model_text, chain_remap, updated_chain_map, temp_model, temp_native)
    """
    # Load and preprocess model
    if model_path.suffix == ".cif":
        model_text, cif_chain_remap = _cif_to_pdb_text(model_path)
    else:
        model_text = model_path.read_text()
        cif_chain_remap = {}

    model_text, normalize_chain_remap = _normalise_pdb_chains(model_text)
    model_text = _strip_hydrogens(model_text)

    # Combine CIF and normalization remappings
    chain_remap = {}
    for orig_id, norm_id in normalize_chain_remap.items():
        actual_orig = cif_chain_remap.get(orig_id, orig_id)
        chain_remap[norm_id] = actual_orig

    # Update chain_map with any remapping
    inverted_remap = {orig: final for final, orig in chain_remap.items()}
    updated_chain_map = {nat: inverted_remap.get(mod, mod) for nat, mod in chain_map.items()}

    # Save preprocessed files
    temp_model = tmp / "model.pdb"
    temp_model.write_text(model_text)
    temp_native = tmp / "native.pdb"
    temp_native.write_text(_strip_hydrogens(native_path.read_text()))

    return model_text, chain_remap, updated_chain_map, temp_model, temp_native


def _verify_chains_exist(
    binder_ordering: tuple,
    merged_binder_id: str,
    merged_model_binder_id: str,
    native_antigen_id: str,
    model_antigen_id: str,
    native_struct,
    model_struct,
) -> bool:
    """Verify that all required chains exist in structures."""
    native_chain_ids = {c.id for c in native_struct.get_chains()}
    model_chain_ids = {c.id for c in model_struct.get_chains()}

    if merged_binder_id not in native_chain_ids:
        LOGGER.debug(f"Ordering {binder_ordering}: Merged binder {merged_binder_id} not in native")
        return False
    if merged_model_binder_id not in model_chain_ids:
        LOGGER.debug(f"Ordering {binder_ordering}: Merged binder {merged_model_binder_id} not in model")
        return False
    if native_antigen_id not in native_chain_ids:
        LOGGER.debug(f"Ordering {binder_ordering}: Antigen {native_antigen_id} not in native")
        return False
    if model_antigen_id not in model_chain_ids:
        LOGGER.debug(f"Ordering {binder_ordering}: Antigen {model_antigen_id} not in model")
        return False
    return True


def _find_interface_key(interfaces: dict, merged_binder_id: str, native_antigen_id: str) -> str | None:
    """Find the interface key in DockQ results."""
    possible_keys = [
        merged_binder_id + native_antigen_id,
        native_antigen_id + merged_binder_id,
    ]
    for key in possible_keys:
        if key in interfaces:
            return key
    return None


def _extract_complex_metrics(
    result: dict,
    selected_antigen_chain: str | None = None,
    used_intended_antigen: bool | None = None,
) -> dict[str, Any]:
    """Extract complex-level metrics from DockQ result."""
    if used_intended_antigen is None:
        used_intended_antigen_value = ""
    elif used_intended_antigen:
        used_intended_antigen_value = "true"
    else:
        used_intended_antigen_value = "false"

    return {
        "complex_dockq": result.get("DockQ"),
        "complex_irmsd": result.get("iRMS") or result.get("iRMSD") or result.get("irms"),
        "complex_lrmsd": result.get("LRMS") or result.get("LRMSD") or result.get("Lrms"),
        "complex_fnat": result.get("fnat"),
        "complex_fnonnat": result.get("fnonnat"),
        "complex_F1": result.get("F1"),
        "complex_clashes": result.get("clashes"),
        "selected_antigen_chain": selected_antigen_chain or "",
        "used_intended_antigen": used_intended_antigen_value,
    }


def _resolve_model_binder_ordering(
    binder_ordering: tuple,
    model_binder_chains_dict: dict,
) -> list[str] | None:
    """Resolve model binder chain ordering from native binder ordering."""
    if not all(ch in model_binder_chains_dict for ch in binder_ordering):
        missing = [ch for ch in binder_ordering if ch not in model_binder_chains_dict]
        LOGGER.debug(
            f"Ordering {binder_ordering}: Skipping - chains {missing} not in model. "
            f"Available: {list(model_binder_chains_dict.keys())}"
        )
        return None
    return [model_binder_chains_dict[ch] for ch in binder_ordering]


def _build_merged_chain_map(
    merged_binder_id: str,
    merged_model_binder_id: str,
    antigen_chains: list,
    updated_chain_map: dict,
) -> dict[str, str]:
    """Build chain map for merged-binder vs antigen interfaces."""
    merged_chain_map = {merged_binder_id: merged_model_binder_id}
    for nat_ag in antigen_chains:
        if nat_ag in updated_chain_map:
            merged_chain_map[nat_ag] = updated_chain_map[nat_ag]
    return merged_chain_map


def _build_antigen_eval_order(
    binder_ordering: tuple,
    antigen_chains: list,
    merged_chain_map: dict,
    intended_antigen_chain: str | None,
) -> list[str]:
    """Build antigen evaluation order: intended first, then remaining available antigens."""
    available_antigens = [ag for ag in antigen_chains if ag in merged_chain_map]
    if not available_antigens:
        LOGGER.debug(f"Ordering {binder_ordering}: No antigen chains available in merged chain map")
        return []

    antigen_eval_order: list[str] = []
    if intended_antigen_chain and intended_antigen_chain in available_antigens:
        antigen_eval_order.append(intended_antigen_chain)
    antigen_eval_order.extend([ag for ag in available_antigens if ag not in antigen_eval_order])
    return antigen_eval_order


def _collect_antigen_candidates(
    interfaces: dict,
    antigen_eval_order: list[str],
    merged_chain_map: dict,
    merged_binder_id: str,
    native_chain_ids: set[str],
    model_chain_ids: set[str],
) -> list[tuple[str, dict, float]]:
    """Collect valid antigen interface candidates with DockQ values."""
    candidates: list[tuple[str, dict, float]] = []
    for native_antigen_id in antigen_eval_order:
        model_antigen_id = merged_chain_map.get(native_antigen_id)
        if not model_antigen_id:
            continue

        if native_antigen_id not in native_chain_ids or model_antigen_id not in model_chain_ids:
            continue

        interface_key = _find_interface_key(interfaces, merged_binder_id, native_antigen_id)
        if interface_key is None:
            continue

        result = interfaces[interface_key]
        dockq_value = result.get("DockQ")
        if dockq_value is None:
            continue

        candidates.append((native_antigen_id, result, float(dockq_value)))

    return candidates


def _select_complex_candidate(
    candidates: list[tuple[str, dict, float]],
    intended_antigen_chain: str | None,
) -> tuple[dict[str, Any], float]:
    """Select intended antigen candidate when present, else best DockQ candidate."""
    if intended_antigen_chain:
        for native_antigen_id, result, dockq_value in candidates:
            if native_antigen_id == intended_antigen_chain:
                return (
                    _extract_complex_metrics(
                        result,
                        selected_antigen_chain=native_antigen_id,
                        used_intended_antigen=True,
                    ),
                    dockq_value,
                )

    selected_antigen_id, best_result, best_dockq = max(candidates, key=lambda t: t[2])
    return (
        _extract_complex_metrics(
            best_result,
            selected_antigen_chain=selected_antigen_id,
            used_intended_antigen=False if intended_antigen_chain else None,
        ),
        best_dockq,
    )


def _try_single_binder_ordering(
    binder_ordering: tuple,
    model_binder_chains_dict: dict,
    antigen_chains: list,
    updated_chain_map: dict,
    temp_model: Path,
    temp_native: Path,
    intended_antigen_chain: str | None,
    load_pdb,
    run_on_all_native_interfaces,
) -> tuple[dict | None, float]:
    """Try a single binder chain ordering and return result and DockQ score."""
    # Reload structures for each permutation
    model_struct = load_pdb(str(temp_model))
    native_struct = load_pdb(str(temp_native))

    # Check if all required chains are available
    model_binder_ordering = _resolve_model_binder_ordering(
        binder_ordering,
        model_binder_chains_dict,
    )
    if model_binder_ordering is None:
        return None, -1.0

    # Merge binder chains
    merged_binder_id = "".join(sorted(binder_ordering))
    merged_model_binder_id = "".join(sorted(model_binder_ordering))

    try:
        model_struct = _merge_chains_in_structure(model_struct, list(model_binder_ordering), merged_model_binder_id)
        native_struct = _merge_chains_in_structure(native_struct, list(binder_ordering), merged_binder_id)
    except Exception as merge_exc:
        LOGGER.debug(f"Failed to merge chains in order {binder_ordering}: {merge_exc}")
        return None, -1.0

    # Build chain map for merged structures
    merged_chain_map = _build_merged_chain_map(
        merged_binder_id,
        merged_model_binder_id,
        antigen_chains,
        updated_chain_map,
    )

    # Build antigen evaluation order: intended first (if available), then others.
    antigen_eval_order = _build_antigen_eval_order(
        binder_ordering,
        antigen_chains,
        merged_chain_map,
        intended_antigen_chain,
    )
    if not antigen_eval_order:
        return None, -1.0

    # Validate binder chains once before DockQ call
    native_chain_ids = {c.id for c in native_struct.get_chains()}
    model_chain_ids = {c.id for c in model_struct.get_chains()}
    if merged_binder_id not in native_chain_ids or merged_model_binder_id not in model_chain_ids:
        LOGGER.debug(
            f"Ordering {binder_ordering}: Missing merged binder chain(s) "
            f"native={merged_binder_id in native_chain_ids} model={merged_model_binder_id in model_chain_ids}"
        )
        return None, -1.0

    # Calculate DockQ
    try:
        interfaces, _ = run_on_all_native_interfaces(model_struct, native_struct, chain_map=merged_chain_map)
    except Exception as dockq_exc:
        LOGGER.debug(f"Ordering {binder_ordering}: DockQ calculation failed: {dockq_exc}")
        return None, -1.0

    # Collect candidate antigen interface results.
    candidates = _collect_antigen_candidates(
        interfaces,
        antigen_eval_order,
        merged_chain_map,
        merged_binder_id,
        native_chain_ids,
        model_chain_ids,
    )

    if not candidates:
        LOGGER.debug(
            f"Ordering {binder_ordering}: No valid merged-binder vs antigen interface found. "
            f"Available interfaces: {list(interfaces.keys())}"
        )
        return None, -1.0

    # Policy: use intended antigen when available; otherwise use best DockQ among antigen chains.
    return _select_complex_candidate(candidates, intended_antigen_chain)


def _build_complex_base_result(binder_chains: list[str], antigen_chains: list[str]) -> dict[str, str]:
    """Build default complex-level result payload."""
    return {
        "binder_chains": "+".join(sorted(binder_chains)) if binder_chains else "",
        "antigen_chains": "+".join(sorted(antigen_chains)) if antigen_chains else "",
        "selected_antigen_chain": "",
        "used_intended_antigen": "",
    }


def _log_complex_antigen_context(
    intended_antigen_chain: str | None,
    antigen_chains: list[str],
    native_stem: str,
) -> None:
    """Log intended-antigen context for complex-level scoring."""
    if intended_antigen_chain:
        if intended_antigen_chain in antigen_chains:
            LOGGER.info(
                "Complex DockQ: intended antigen '%s' provided for %s (available antigens=%s)",
                intended_antigen_chain,
                native_stem,
                "+".join(sorted(antigen_chains)),
            )
        else:
            LOGGER.warning(
                "Complex DockQ: intended antigen '%s' not present in available antigens %s for %s; "
                "falling back to best available antigen interface",
                intended_antigen_chain,
                "+".join(sorted(antigen_chains)),
                native_stem,
            )
        return

    if len(antigen_chains) > 1:
        LOGGER.warning(
            "Complex DockQ: multiple antigen chains (%s) but no intended antigen provided for %s; "
            "using best available antigen interface",
            "+".join(sorted(antigen_chains)),
            native_stem,
        )


def _select_single_binder_interface(
    interfaces: dict[str, Any],
    binder_chain: str,
    antigen_chains: list[str],
    intended_antigen_chain: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Select the best interface for a single binder chain.

    Parameters
    ----------
    interfaces : dict
        Interface-level DockQ results keyed by chain pair
    binder_chain : str
        The single binder chain ID
    antigen_chains : list of str
        Available antigen chain IDs
    intended_antigen_chain : str or None
        Optional intended antigen chain ID to prioritize

    Returns
    -------
    tuple of (str or None, dict or None)
        (selected_antigen_chain, interface_metrics)
    """

    def get_interface(antigen: str) -> dict[str, Any] | None:
        """Get interface metrics for antigen, trying both chain orderings."""
        for key in [binder_chain + antigen, antigen + binder_chain]:
            if key in interfaces:
                return interfaces[key]
        return None

    # Try intended antigen first if provided
    if intended_antigen_chain in antigen_chains:
        interface = get_interface(intended_antigen_chain)
        if interface:
            return intended_antigen_chain, interface

    # Find best available interface by DockQ score
    best_dockq = -1.0
    selected_antigen = None
    selected_interface = None

    for antigen_chain in antigen_chains:
        interface = get_interface(antigen_chain)
        if interface:
            dockq_score = interface.get("DockQ")
            is_better_score = dockq_score is not None and dockq_score > best_dockq
            if is_better_score:
                best_dockq = dockq_score
                selected_antigen = antigen_chain
                selected_interface = interface

    return selected_antigen, selected_interface


def _build_single_binder_result(
    base_result: dict[str, str],
    interface: dict[str, Any],
    selected_antigen: str | None,
    intended_antigen_chain: str | None,
) -> dict[str, Any]:
    """Build result dict from single binder interface metrics.

    Parameters
    ----------
    base_result : dict
        Base result with chain labels
    interface : dict
        Interface metrics from DockQ
    selected_antigen : str or None
        The antigen chain that was selected
    intended_antigen_chain : str or None
        The intended antigen chain (if any)

    Returns
    -------
    dict
        Complete result dict with complex-level metrics
    """
    result = base_result.copy()

    matched_intended = intended_antigen_chain and selected_antigen == intended_antigen_chain
    has_intended = intended_antigen_chain is not None

    if matched_intended:
        used_intended_antigen = "true"
    elif has_intended:
        used_intended_antigen = "false"
    else:
        used_intended_antigen = ""

    result.update(
        {
            "complex_dockq": interface.get("DockQ"),
            "complex_irmsd": interface.get("iRMSD"),
            "complex_lrmsd": interface.get("LRMSD"),
            "complex_fnat": interface.get("fnat"),
            "complex_fnonnat": interface.get("fnonnat"),
            "complex_F1": interface.get("F1"),
            "complex_clashes": interface.get("clashes"),
            "selected_antigen_chain": selected_antigen or "",
            "used_intended_antigen": used_intended_antigen,
        }
    )
    return result


def _handle_single_binder_case(
    model_path: Path,
    native_path: Path,
    chain_map: dict[str, str],
    binder_chains: list[str],
    antigen_chains: list[str],
    intended_antigen_chain: str | None,
    base_result: dict[str, str],
) -> dict[str, Any]:
    """Handle the single binder chain case.

    Parameters
    ----------
    model_path : Path
        Predicted structure file
    native_path : Path
        Native reference structure
    chain_map : dict
        Chain mapping
    binder_chains : list of str
        Binder chain IDs (should have length 1)
    antigen_chains : list of str
        Antigen chain IDs
    intended_antigen_chain : str or None
        Optional intended antigen
    base_result : dict
        Base result with chain labels

    Returns
    -------
    dict
        Complex-level result or base_result if computation fails
    """
    LOGGER.debug("Single binder chain - complex score same as interface score")
    try:
        interfaces, _ = run_dockq(model_path, native_path, chain_map)
        binder_chain = binder_chains[0]

        selected_antigen, selected_interface = _select_single_binder_interface(
            interfaces,
            binder_chain,
            antigen_chains,
            intended_antigen_chain,
        )

        if selected_interface is not None:
            return _build_single_binder_result(
                base_result,
                selected_interface,
                selected_antigen,
                intended_antigen_chain,
            )
        else:
            LOGGER.warning(f"Single binder {binder_chain}: No valid interface found with antigens {antigen_chains}")
            return base_result

    except Exception as exc:
        LOGGER.warning(f"Failed to compute single-binder complex score: {exc}", exc_info=True)
        return base_result


def _evaluate_binder_permutations(
    binder_permutations: list[tuple[str, ...]],
    model_binder_chains_dict: dict[str, str],
    antigen_chains: list[str],
    updated_chain_map: dict[str, str],
    temp_model: Path,
    temp_native: Path,
    intended_antigen_chain: str | None,
    load_pdb,
    run_on_all_native_interfaces,
) -> tuple[dict[str, Any] | None, float, tuple[str, ...] | None]:
    """Evaluate all binder permutations and return the best-scoring result."""
    best_result: dict[str, Any] | None = None
    best_dockq = -1.0
    best_ordering: tuple[str, ...] | None = None

    for binder_ordering in binder_permutations:
        result, current_dockq = _try_single_binder_ordering(
            binder_ordering,
            model_binder_chains_dict,
            antigen_chains,
            updated_chain_map,
            temp_model,
            temp_native,
            intended_antigen_chain,
            load_pdb=load_pdb,
            run_on_all_native_interfaces=run_on_all_native_interfaces,
        )
        if result and current_dockq > best_dockq:
            best_dockq = current_dockq
            best_ordering = binder_ordering
            best_result = result

    return best_result, best_dockq, best_ordering


def _compute_best_complex_result(
    model_path: Path,
    native_path: Path,
    chain_map: dict[str, str],
    binder_chains: list[str],
    antigen_chains: list[str],
    intended_antigen_chain: str | None,
    load_pdb,
    run_on_all_native_interfaces,
) -> tuple[dict[str, Any] | None, float, tuple[str, ...] | None, int]:
    """Compute best complex-level result across binder permutations."""
    import itertools

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _, _, updated_chain_map, temp_model, temp_native = _preprocess_structures_for_complex_dockq(
            model_path,
            native_path,
            chain_map,
            tmp,
        )

        model_binder_chains_dict = {ch: updated_chain_map[ch] for ch in binder_chains if ch in updated_chain_map}
        model_antigen_chains = [updated_chain_map[ch] for ch in antigen_chains if ch in updated_chain_map]
        if not model_binder_chains_dict or not model_antigen_chains:
            return None, -1.0, None, 0

        binder_permutations = list(itertools.permutations(binder_chains))
        best_result, best_dockq, best_ordering = _evaluate_binder_permutations(
            binder_permutations,
            model_binder_chains_dict,
            antigen_chains,
            updated_chain_map,
            temp_model,
            temp_native,
            intended_antigen_chain,
            load_pdb=load_pdb,
            run_on_all_native_interfaces=run_on_all_native_interfaces,
        )
        return best_result, best_dockq, best_ordering, len(binder_permutations)


def _log_complex_result_selection(
    best_dockq: float,
    best_ordering: tuple[str, ...] | None,
    permutation_count: int,
    best_result: dict[str, Any],
    intended_antigen_chain: str | None,
    native_stem: str,
) -> None:
    """Log summary of selected complex-level result."""
    selected_antigen_chain = best_result.get("selected_antigen_chain", "")
    used_intended_antigen = best_result.get("used_intended_antigen", "")
    LOGGER.debug(
        f"Best complex DockQ = {best_dockq:.4f} with binder ordering {best_ordering} "
        f"(tried {permutation_count} permutations)"
    )
    if used_intended_antigen == "false":
        LOGGER.warning(
            "Complex DockQ fallback: intended antigen '%s' not used; selected '%s' for %s",
            intended_antigen_chain,
            selected_antigen_chain,
            native_stem,
        )
    elif used_intended_antigen == "true":
        LOGGER.info(
            "Complex DockQ used intended antigen '%s' for %s",
            selected_antigen_chain,
            native_stem,
        )
    elif selected_antigen_chain:
        LOGGER.info(
            "Complex DockQ selected antigen '%s' for %s",
            selected_antigen_chain,
            native_stem,
        )


def run_complex_level_dockq(
    model_path: Path,
    native_path: Path,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
    intended_antigen_chain: str | None = None,
) -> dict[str, Any] | None:
    """Compute complex-level DockQ by treating multiple binder chains as one unit.

    For antibody-antigen complexes, this treats H+L chains as a single "binder"
    entity and computes DockQ against the antigen, giving a single score that
    captures the quality of the complete antibody-antigen interaction.

    Parameters
    ----------
    model_path : Path
        Predicted structure file (.pdb or .cif)
    native_path : Path
        Native reference structure (.pdb)
    chain_map : dict
        {native_chain: model_chain} mapping
    native_chain_roles : dict
        {native_chain: role} mapping where role in {"binder", "antigen"}
    intended_antigen_chain : str or None
        Optional native-chain ID for the intended antigen target. When provided,
        complex metrics are taken from the merged-binder vs this antigen
        interface if available.

    Returns
    -------
    dict or None
        Complex-level DockQ metrics, or None if complex scoring is not applicable
        (e.g., no clear binder/antigen separation)
    """
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError:
        LOGGER.warning("DockQ not installed, skipping complex-level scoring")
        return None

    # Identify binder and antigen chains in native structure
    binder_chains = [ch for ch, role in native_chain_roles.items() if role == "binder"]
    antigen_chains = [ch for ch, role in native_chain_roles.items() if role == "antigen"]

    # Always return chain labels even if we can't compute complex DockQ
    base_result = _build_complex_base_result(binder_chains, antigen_chains)

    # Need at least one chain of each type
    if not binder_chains or not antigen_chains:
        LOGGER.debug("Cannot compute complex-level DockQ: no clear binder/antigen split")
        return base_result

    _log_complex_antigen_context(
        intended_antigen_chain,
        antigen_chains,
        native_path.stem,
    )

    # If only one binder chain, complex-level = interface-level (no merging needed)
    if len(binder_chains) == 1:
        return _handle_single_binder_case(
            model_path,
            native_path,
            chain_map,
            binder_chains,
            antigen_chains,
            intended_antigen_chain,
            base_result,
        )

    try:
        best_result, best_dockq, best_ordering, permutation_count = _compute_best_complex_result(
            model_path,
            native_path,
            chain_map,
            binder_chains,
            antigen_chains,
            intended_antigen_chain,
            load_pdb=load_PDB,
            run_on_all_native_interfaces=run_on_all_native_interfaces,
        )
        if best_result is None:
            if permutation_count:
                LOGGER.warning(f"No valid complex DockQ result found after trying {permutation_count} chain orderings")
            return base_result

        _log_complex_result_selection(
            best_dockq,
            best_ordering,
            permutation_count,
            best_result,
            intended_antigen_chain,
            native_path.stem,
        )
        result_dict = base_result.copy()
        result_dict.update(best_result)
        return result_dict

    except Exception as exc:
        LOGGER.warning(f"Complex-level DockQ calculation failed: {exc}", exc_info=True)
        return base_result


def run_dockq_with_complex_score(
    model_path: Path,
    native_path: Path,
    chain_map: dict[str, str],
    native_chain_roles: dict[str, str],
    intended_antigen_chain: str | None = None,
) -> tuple[dict[str, dict[str, Any]], float, dict[str, Any] | None]:
    """Run both per-interface and complex-level DockQ scoring.

    This is a wrapper that combines standard per-interface DockQ with
    complex-level scoring where applicable.

    Parameters
    ----------
    model_path : Path
        Predicted structure file
    native_path : Path
        Native reference structure
    chain_map : dict
        {native_chain: model_chain} mapping
    native_chain_roles : dict
        {native_chain: role} mapping (from infer_chain_roles)
    intended_antigen_chain : str or None
        Optional native-chain ID for the intended antigen target.

    Returns
    -------
    tuple
        (interfaces, global_dockq, complex_metrics) where:
        - interfaces: dict mapping interface keys to per-interface metrics
        - global_dockq: average DockQ across all interfaces
        - complex_metrics: dict of complex-level metrics, or None if not applicable
    """
    # Run standard per-interface DockQ
    interfaces, global_dockq = run_dockq(model_path, native_path, chain_map)

    # Run complex-level DockQ
    complex_metrics = run_complex_level_dockq(
        model_path,
        native_path,
        chain_map,
        native_chain_roles,
        intended_antigen_chain=intended_antigen_chain,
    )

    return interfaces, global_dockq, complex_metrics
