"""Tests for the DockQ-predictor data layer: chain mapping, graph building, dataset, sampler.

The graph tests drive the REAL builder on small synthetic two-chain PDBs rather than
mocking it, so the k-NN + cross-chain edge union, the interface mask and the
metadata the feature builders consume are all exercised end to end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from confidock.dockq_predictor.data import chain_mapping_cif as cmc
from confidock.dockq_predictor.data.chain_mapping import parse_chain_string
from confidock.dockq_predictor.data.datamodule import (
    DockQDataModule,
    FilteredDockQDataset,
    GroupedComplexBatchSampler,
)
from confidock.dockq_predictor.data.dataset import (
    FEATURE_VERSION,
    DockQDataset,
    _native_id_from_path,
    create_data_splits,
)
from confidock.dockq_predictor.data.feature_utils import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from confidock.dockq_predictor.data.graph_builder import (
    _knn_edges_from_distances,
    build_graph,
    fix_esmfold2_cif,
    identify_antibody_chains,
)

# GLY has no side chain, so N/CA/C/O is its complete heavy-atom set -- a residue
# written with just those four atoms is chemically complete rather than truncated.
_ATOMS = (("N", (-1.2, 0.3, 0.0)), ("CA", (0.0, 0.0, 0.0)), ("C", (1.2, 0.4, 0.0)), ("O", (1.3, 1.6, 0.0)))


def _pdb_text(chains: dict, plddt: float = 80.0) -> str:
    """Write a minimal multi-chain poly-glycine PDB.

    ``chains`` maps chain_id -> list of CA positions; each residue gets a full
    N/CA/C/O backbone offset from its CA.
    """
    lines = []
    serial = 1
    for chain_id, ca_positions in chains.items():
        for res_index, ca in enumerate(ca_positions, start=1):
            for name, offset in _ATOMS:
                x, y, z = (ca[i] + offset[i] for i in range(3))
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} GLY {chain_id}{res_index:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           {name[0]}"
                )
                serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _two_chain_pdb(path: Path, gap: float = 5.0, n_per_chain: int = 8) -> Path:
    """Two 8-residue chains H and A, running parallel ``gap`` Angstroms apart."""
    chains = {
        "H": [(3.8 * i, 0.0, 0.0) for i in range(n_per_chain)],
        "A": [(3.8 * i, gap, 0.0) for i in range(n_per_chain)],
    }
    path.write_text(_pdb_text(chains))
    return path


def _entry(pdb_path: Path, dockq: float = 0.5, model_source: str = "boltz2") -> dict:
    """Build a manifest entry exactly as ``manifest_df_to_entries`` produces one."""
    return {"pdb_path": str(pdb_path), "model_source": model_source, "chain_str": "HA", "dockq": dockq}


# ---------------------------------------------------------------------------
# parse_chain_string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chain_str", "expected"),
    [
        ("HLA", {"H": "VH", "L": "VL", "A": "antigen"}),
        ("HcLcA", {"Hc": "VH", "Lc": "VL", "A": "antigen"}),
        ("KMB", {"K": "VL", "M": "antigen", "B": "antigen"}),  # kappa light chain
        ("AxB", {"A": "antigen", "x": "antigen", "B": "antigen"}),  # nanobody-free complex
        ("", {}),
    ],
)
def test_parse_chain_string_follows_the_sabdab_convention(chain_str: str, expected: dict) -> None:
    """H/Hc -> VH, L/K (and Lc/Kc) -> VL, anything else -> antigen."""
    assert parse_chain_string(chain_str, "x.pdb") == expected


def test_parse_chain_string_prefers_the_two_character_chain_id() -> None:
    """Treat "Hc" as ONE chain, not H followed by an antigen chain c."""
    assert parse_chain_string("HcA", "x.pdb") == {"Hc": "VH", "A": "antigen"}


# ---------------------------------------------------------------------------
# SAbDab JSON sidecar lookup
# ---------------------------------------------------------------------------


def test_load_sabdab_json_strips_the_decoy_suffixes(tmp_path: Path, monkeypatch) -> None:
    """Every decoy of a complex must resolve to the complex's single JSON."""
    (tmp_path / "1abc_HLA.json").write_text(json.dumps({"h_seq": "GG"}))
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    for filename in ("1abc_HLA.cif", "1abc_HLA_sample_0.cif", "1abc_HLA_seed42_sample0.pdb"):
        assert cmc.load_sabdab_json(f"/some/dir/{filename}") == {"h_seq": "GG"}


def test_load_sabdab_json_returns_none_when_absent(tmp_path: Path, monkeypatch) -> None:
    """A missing sidecar is a normal outcome, not an error."""
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    assert cmc.load_sabdab_json("/some/dir/9zzz_HLA.cif") is None


def test_identify_chains_from_json_matches_chains_by_length(tmp_path: Path, monkeypatch) -> None:
    """CIF chains are numeric, so typing falls back to sequence-length matching."""
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    (tmp_path / "1abc_HLA.json").write_text(json.dumps({"h_seq": "G" * 120, "l_seq": "G" * 108, "t_seq": "G" * 50}))
    types = cmc.identify_chains_from_json(f"{tmp_path}/1abc_HLA.cif", {"1": 121, "2": 108, "3": 50, "4": 7})
    # Chain 1 is within the +-2 tolerance of h_seq; chain 4 matches nothing and
    # defaults to antigen rather than being dropped.
    assert types == {"1": "VH", "2": "VL", "3": "antigen", "4": "antigen"}


def test_identify_chains_from_json_is_empty_without_metadata(tmp_path: Path, monkeypatch) -> None:
    """No JSON means no typing claim at all (the caller then tries ANARCI)."""
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    assert cmc.identify_chains_from_json(f"{tmp_path}/9zzz_HLA.cif", {"1": 120}) == {}


def test_align_ref_positions_handles_terminal_truncation() -> None:
    """A structure missing its first residues must still get its CDRs labelled."""
    pos_map = cmc._align_ref_positions_to_structure("ACDEFGHIK", "DEFGHIK")
    assert pos_map[3] == 0  # ref position 3 (D) is the structure's first residue
    assert pos_map[9] == 6
    assert 1 not in pos_map  # the unmodelled residues simply have no mapping


# ---------------------------------------------------------------------------
# k-NN edges
# ---------------------------------------------------------------------------


def test_knn_edges_picks_the_k_closest_neighbours_pointing_at_the_query() -> None:
    """Edges run neighbour -> query (PyG's source_to_target flow), self excluded."""
    # Four points on a line at x = 0, 1, 2, 10.
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
    edge_index = _knn_edges_from_distances(torch.cdist(pos, pos), 2)

    edges = {(int(s), int(d)) for s, d in edge_index.t()}
    assert edge_index.shape == (2, 8)  # 4 nodes x k=2
    assert not any(s == d for s, d in edges)  # no self-loops
    # Node 0's two closest neighbours are 1 and 2 -- as sources of an edge into 0.
    assert {s for s, d in edges if d == 0} == {1, 2}
    # The far node 3 still gets its k neighbours; nothing else is forced to pick it.
    assert {s for s, d in edges if d == 3} == {1, 2}


def test_knn_edges_caps_k_at_the_number_of_other_nodes() -> None:
    """A k above N-1 degrades to a fully connected graph; a single node gets no edges."""
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    assert _knn_edges_from_distances(torch.cdist(pos, pos), 16).shape == (2, 6)  # 3 x (3-1)

    single = torch.zeros(1, 1)
    assert _knn_edges_from_distances(single, 4).shape == (2, 0)
    assert _knn_edges_from_distances(single, 4).dtype == torch.long


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def test_build_graph_makes_one_node_per_residue_with_full_metadata(tmp_path: Path) -> None:
    """The builder produces topology + the metadata the feature builders consume."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    data = build_graph(str(pdb), interface_cutoff=8.0, knn_k=4, chain_types={"H": "VH", "A": "antigen"})

    assert data.num_nodes == 16
    assert data.pos.shape == (16, 3)
    assert data.chain_id.tolist() == [0] * 8 + [1] * 8
    assert data.is_antibody.tolist() == [True] * 8 + [False] * 8
    assert data.is_heavy.tolist() == [True] * 8 + [False] * 8
    assert data.plddt.shape == (16,)
    assert torch.allclose(data.plddt, torch.full((16,), 80.0))
    assert len(data.residues) == 16
    assert data.residues[0]["backbone_coords"].keys() == {"N", "CA", "C", "O"}
    assert data.residues[0]["heavy_atom_coords"].shape == (4, 3)


def test_build_graph_edges_union_knn_with_every_cross_chain_pair_within_12a(tmp_path: Path) -> None:
    """Both edge sources are present, deduplicated, and symmetric."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb", gap=5.0)
    data = build_graph(str(pdb), knn_k=4, chain_types={"H": "VH", "A": "antigen"})

    edges = {(int(s), int(d)) for s, d in data.edge_index.t()}
    assert len(edges) == data.edge_index.shape[1]  # torch.unique deduplicated
    # Residue 0 of H and residue 0 of A are 5A apart -> a cross-chain edge exists
    # in both directions even though k-NN alone might not pick it.
    assert (0, 8) in edges and (8, 0) in edges
    # A cross-chain pair far beyond 12A is not connected.
    assert (0, 15) not in edges


def test_build_graph_interface_mask_respects_the_cutoff(tmp_path: Path) -> None:
    """is_interface = has a cross-chain partner within the cutoff -- nothing else."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb", gap=5.0)
    kwargs = dict(knn_k=4, chain_types={"H": "VH", "A": "antigen"})

    assert bool(build_graph(str(pdb), interface_cutoff=6.0, **kwargs).is_interface.all())
    # Pull the chains apart: at a 4A cutoff nothing is within reach any more.
    assert not bool(build_graph(str(pdb), interface_cutoff=4.0, **kwargs).is_interface.any())


def test_build_graph_labels_cdrs_from_the_json_sidecar(tmp_path: Path, monkeypatch) -> None:
    """CDR labels come from h_cdr/l_cdr 1-based positions, mapped by residue order."""
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    (tmp_path / "1abc_HA.json").write_text(json.dumps({"h_seq": "G" * 8, "h_cdr": {"1": [2], "2": [4], "3": [6, 7]}}))
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    data = build_graph(str(pdb), knn_k=4, chain_types={"H": "VH", "A": "antigen"})

    # 1-based JSON positions 2/4/6/7 -> 0-based nodes 1/3/5/6, labels H1/H2/H3/H3.
    assert data.cdr_label.tolist() == [0, 1, 0, 2, 0, 3, 3, 0] + [0] * 8


def test_build_graph_ignores_chain_types_that_do_not_match_the_structure(tmp_path: Path, monkeypatch) -> None:
    """A manifest chain_str for symbolic chains must not be applied to real ones.

    With no JSON to fall back on either, every chain defaults to antigen -- which is
    what keeps a mismatched chain string from mislabelling the heavy chain.
    """
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    data = build_graph(str(pdb), knn_k=4, chain_types={"X": "VH", "Y": "antigen"})
    assert not bool(data.is_antibody.any())


# ---------------------------------------------------------------------------
# Per-source file fixups reached from build_graph
# ---------------------------------------------------------------------------


def _minimal_cif(path: Path, with_occupancy: bool) -> Path:
    """Write a two-residue, two-chain mmCIF, optionally missing the occupancy column.

    ESMFold2 writes the latter; ``MMCIFParser`` raises ``KeyError('occupancy')``
    on it, which is the only route into ``fix_esmfold2_cif``.
    """
    columns = [
        "_atom_site.group_PDB",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    if with_occupancy:
        columns.append("_atom_site.occupancy")
    # fix_esmfold2_cif inserts the occupancy column immediately before
    # _atom_site.id, so that header must come last.
    columns.append("_atom_site.id")

    rows = []
    serial = 1
    for chain, y in (("H", 0.0), ("A", 5.0)):
        for res_index in (1, 2):
            for name, offset in _ATOMS:
                x, yy, z = (3.8 * res_index + offset[0], y + offset[1], offset[2])
                fields = ["ATOM", name[0], name, ".", "GLY", chain, str(res_index), "?"]
                fields += [f"{x:.3f}", f"{yy:.3f}", f"{z:.3f}", "80.00", str(res_index), chain, "1"]
                if with_occupancy:
                    fields.append("1.0")
                fields.append(str(serial))
                rows.append(" ".join(fields))
                serial += 1
    path.write_text("data_test\n#\nloop_\n" + "\n".join(columns) + "\n" + "\n".join(rows) + "\n#\n")
    return path


def test_esmfold2_cif_fixup_adds_the_missing_occupancy_column(tmp_path: Path) -> None:
    """One occupancy header plus a 1.0 per atom, inserted before the id field."""
    fixed = Path(fix_esmfold2_cif(str(_minimal_cif(tmp_path / "1abc_HA.cif", with_occupancy=False))))
    text = fixed.read_text()
    assert text.count("_atom_site.occupancy") == 1
    atom_lines = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) == 16
    assert all(line.split()[-2] == "1.0" for line in atom_lines)


def test_esmfold2_cif_fixup_is_a_no_op_when_occupancy_is_present(tmp_path: Path) -> None:
    """A conformant CIF is returned unchanged -- and by path, not a temp copy."""
    cif = _minimal_cif(tmp_path / "1abc_HA.cif", with_occupancy=True)
    assert fix_esmfold2_cif(str(cif)) == str(cif)


@pytest.mark.parametrize("with_occupancy", [True, False])
def test_build_graph_reads_a_cif_with_or_without_occupancy(tmp_path: Path, monkeypatch, with_occupancy: bool) -> None:
    """The esmfold2 path end to end: the fixup must yield the same graph.

    Both spellings describe the same 4 residues, so the recovered topology must be
    identical -- occupancy is metadata the builder never uses.
    """
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    cif = _minimal_cif(tmp_path / f"1abc_HA_{with_occupancy}.cif", with_occupancy=with_occupancy)
    data = build_graph(str(cif), interface_cutoff=6.0, knn_k=4, chain_types={"H": "VH", "A": "antigen"})
    assert data.num_nodes == 4
    assert data.chain_id.tolist() == [0, 0, 1, 1]
    assert data.is_antibody.tolist() == [True, True, False, False]


def test_build_graph_fixes_and_remaps_multi_char_boltz2_chains(tmp_path: Path, monkeypatch) -> None:
    """Chain types are keyed on the ORIGINAL ids, so they must be remapped.

    ``fix_boltz2_pdb`` renames Hc -> A and A -> B; without the remap the provided
    ``{"Hc": "VH"}`` would no longer match any chain and the heavy chain would be
    silently demoted to antigen (see the mismatch test above).
    """
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    # Take a well-formed two-chain PDB and reproduce the boltz2 malformation: widen
    # chain H's id to "Hc" by INSERTING a column, pushing the rest of that line one
    # place right (chain A's lines stay correctly aligned, as boltz2 mixes widths).
    plain = _pdb_text({"H": [(3.8 * i, 0.0, 0.0) for i in range(2)], "A": [(3.8 * i, 5.0, 0.0) for i in range(2)]})
    widened = [line[:21] + "Hc" + line[22:] if line[21:22] == "H" else line for line in plain.splitlines()]
    pdb = tmp_path / "1abc_HcA.pdb"
    pdb.write_text("\n".join(widened) + "\n")

    data = build_graph(str(pdb), interface_cutoff=6.0, knn_k=4, chain_types={"Hc": "VH", "A": "antigen"})
    assert data.num_nodes == 4
    assert data.is_antibody.tolist() == [True, True, False, False]
    assert data.is_heavy.tolist() == [True, True, False, False]
    assert data.chain_types == {"A": "VH", "B": "antigen"}  # renamed, not dropped


def test_build_graph_falls_back_to_anarci_when_there_is_no_json(tmp_path: Path, monkeypatch, capsys) -> None:
    """Last resort when neither the manifest nor a JSON sidecar can type a chain.

    ANARCI is not installed in this environment, so this exercises the
    FileNotFoundError branch: an 8-residue chain is far too short to look like an
    antibody, so both chains come back as antigen rather than crashing.
    """
    monkeypatch.setenv("CONFIDOCK_SABDAB_JSON_DIR", str(tmp_path))
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    data = build_graph(str(pdb), interface_cutoff=6.0, knn_k=4)
    assert "using ANARCI" in capsys.readouterr().out
    assert not bool(data.is_antibody.any())


def test_anarci_length_heuristic_flags_antibody_sized_chains(tmp_path: Path, monkeypatch) -> None:
    """Without ANARCI, a 100-140 residue chain is provisionally "antibody"."""
    monkeypatch.setattr(
        "confidock.dockq_predictor.data.graph_builder.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ANARCI")),
    )
    pdb = tmp_path / "1abc_HA.pdb"
    pdb.write_text(_pdb_text({"H": [(3.8 * i, 0.0, 0.0) for i in range(120)], "A": [(0.0, 40.0, 0.0)]}))
    assert identify_antibody_chains(str(pdb)) == {"H": "antibody", "A": "antigen"}


def test_anarci_output_is_parsed_when_it_runs(tmp_path: Path, monkeypatch) -> None:
    """The real branch, with ANARCI's stdout stubbed: VH / VL / antigen."""

    class _Result:
        returncode = 0

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    outputs = iter(["# Domain 1 of 1\nH|chain_H|\n", "# Domain 1 of 1\nL|chain_A|\n"])

    monkeypatch.setattr(
        "confidock.dockq_predictor.data.graph_builder.subprocess.run",
        lambda *a, **k: _Result(next(outputs)),
    )
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    assert identify_antibody_chains(str(pdb)) == {"H": "VH", "A": "VL"}


def test_anarci_failure_defaults_a_chain_to_antigen(tmp_path: Path, monkeypatch) -> None:
    """A non-zero exit must not propagate; an untyped chain is an antigen."""

    class _Failed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr("confidock.dockq_predictor.data.graph_builder.subprocess.run", lambda *a, **k: _Failed())
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    assert identify_antibody_chains(str(pdb)) == {"H": "antigen", "A": "antigen"}


# ---------------------------------------------------------------------------
# _native_id_from_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/d/7y8j_HLA_sample_0.cif", "7y8j_HLA"),
        ("/d/7y8j_HLA.cif", "7y8j_HLA"),  # extension stripped BEFORE tokenising
        ("/d/7jtf_ABE_seed42_sample0.pdb", "7jtf_ABE"),
        ("7y8j_HLA_sample_3.pdb", "7y8j_HLA"),
    ],
)
def test_native_id_is_pdb_id_plus_chains(path: str, expected: str) -> None:
    """Keyed on "{pdb_id}_{chains}" so alternative binding modes stay distinct."""
    assert _native_id_from_path(path) == expected


def test_native_id_separates_binding_modes_of_one_pdb_id() -> None:
    """The reason it is not just the pdb_id: ~60 pdb_ids have several modes."""
    assert _native_id_from_path("/d/1a3r_HLP_sample_0.cif") != _native_id_from_path("/d/1a3r_ABC_sample_0.cif")


# ---------------------------------------------------------------------------
# DockQDataset
# ---------------------------------------------------------------------------


def test_dataset_get_produces_a_labelled_featurised_graph(tmp_path: Path) -> None:
    """One entry in, one collatable 18/96-dim labelled graph out."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    dataset = DockQDataset([_entry(pdb, dockq=0.42)], interface_cutoff=6.0, knn_k=4)
    data = dataset[0]

    assert data.x.shape == (16, NODE_FEATURE_DIM)
    assert data.edge_attr.shape == (data.edge_index.shape[1], EDGE_FEATURE_DIM)
    assert data.y_dockq.tolist() == pytest.approx([0.42])
    assert data.native_id == "1abc_HA"
    assert data.model_source == "boltz2"
    # The Bio.PDB-derived attributes are dropped: they would bloat the cache and
    # cannot be collated into a batch.
    assert not hasattr(data, "residues")
    assert not hasattr(data, "chain_types")


def test_dataset_applies_the_manifest_chain_string(tmp_path: Path) -> None:
    """chain_str="HA" types chain H as the heavy chain without needing a JSON."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    data = DockQDataset([_entry(pdb)], interface_cutoff=6.0, knn_k=4)[0]
    is_heavy_col = 8  # cdr(7) + is_antibody(1) -> is_heavy
    assert data.x[:8, is_heavy_col].tolist() == pytest.approx([1.0] * 8)
    assert data.x[8:, is_heavy_col].tolist() == pytest.approx([0.0] * 8)


def test_dataset_returns_none_for_an_unparseable_structure(tmp_path: Path) -> None:
    """One malformed file must not kill a run (FilteredDockQDataset resamples)."""
    bad = tmp_path / "9zzz_HA.cif"
    bad.write_text("not a structure at all\n")
    assert DockQDataset([_entry(bad)], knn_k=4).get(0) is None


def test_dataset_memory_cache_returns_the_identical_object(tmp_path: Path) -> None:
    """cache_graphs keeps built graphs in RAM, so a second get is the same object."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    dataset = DockQDataset([_entry(pdb)], interface_cutoff=6.0, knn_k=4, cache_graphs=True)
    assert dataset.get(0) is dataset.get(0)

    uncached = DockQDataset([_entry(pdb)], interface_cutoff=6.0, knn_k=4, cache_graphs=False)
    assert uncached.get(0) is not uncached.get(0)


def test_dataset_disk_cache_round_trips_an_identical_graph(tmp_path: Path) -> None:
    """A cache hit must be indistinguishable from a fresh build."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    cache_dir = tmp_path / "graph_cache"
    kwargs = dict(interface_cutoff=6.0, knn_k=4, cache_graphs=False, graph_cache_dir=str(cache_dir))

    fresh = DockQDataset([_entry(pdb)], **kwargs).get(0)
    assert len(list(cache_dir.glob("*.pt"))) == 1
    assert not list(cache_dir.glob("*.tmp.*"))  # the temp file was renamed, not left behind

    loaded = DockQDataset([_entry(pdb)], **kwargs).get(0)
    assert torch.equal(loaded.x, fresh.x)
    assert torch.equal(loaded.edge_attr, fresh.edge_attr)
    assert torch.equal(loaded.edge_index, fresh.edge_index)
    assert loaded.native_id == fresh.native_id


def test_dataset_treats_a_corrupt_cache_file_as_a_miss(tmp_path: Path) -> None:
    """An interrupted write must cause a rebuild, not a crash."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    cache_dir = tmp_path / "graph_cache"
    kwargs = dict(interface_cutoff=6.0, knn_k=4, cache_graphs=False, graph_cache_dir=str(cache_dir))
    dataset = DockQDataset([_entry(pdb)], **kwargs)
    Path(dataset._disk_cache_path(dataset.entries[0])).write_bytes(b"truncated")

    data = dataset.get(0)
    assert data.x.shape == (16, NODE_FEATURE_DIM)


def test_dataset_re_derives_native_id_on_a_cache_hit(tmp_path: Path) -> None:
    """native_id is not in the cache key, so a stale stored value must be overwritten."""
    pdb = _two_chain_pdb(tmp_path / "1abc_HA.pdb")
    cache_dir = tmp_path / "graph_cache"
    kwargs = dict(interface_cutoff=6.0, knn_k=4, cache_graphs=False, graph_cache_dir=str(cache_dir))
    dataset = DockQDataset([_entry(pdb)], **kwargs)

    data = dataset.get(0)
    data.native_id = "1abc_HA.pdb"  # the value graphs cached before the fix carried
    torch.save(data, dataset._disk_cache_path(dataset.entries[0]))
    assert dataset.get(0).native_id == "1abc_HA"


@pytest.mark.parametrize(
    "changed",
    [
        {"interface_cutoff": 8.0},
        {"knn_k": 32},
        {"entry_override": {"model_source": "protenix"}},
        {"entry_override": {"chain_str": "HLA"}},
    ],
)
def test_disk_cache_key_changes_with_every_graph_determining_input(tmp_path: Path, changed: dict) -> None:
    """Stale graphs are never silently reused when the recipe changes."""
    base_kwargs = dict(interface_cutoff=6.0, knn_k=16, graph_cache_dir=str(tmp_path / "c"))
    entry = _entry(tmp_path / "1abc_HA.pdb")

    entry_override = changed.pop("entry_override", {})
    reference = DockQDataset([entry], **base_kwargs)
    variant = DockQDataset([{**entry, **entry_override}], **{**base_kwargs, **changed})
    assert reference._disk_cache_path(reference.entries[0]) != variant._disk_cache_path(variant.entries[0])


def test_disk_cache_key_is_stable_for_an_unchanged_recipe(tmp_path: Path) -> None:
    """The flip side: two runs of the same config must share the warmed cache."""
    entry = _entry(tmp_path / "1abc_HA.pdb")
    kwargs = dict(interface_cutoff=6.0, knn_k=16, graph_cache_dir=str(tmp_path / "c"))
    first = DockQDataset([entry], **kwargs)
    second = DockQDataset([dict(entry)], **kwargs)
    assert first._disk_cache_path(entry) == second._disk_cache_path(entry)


def test_disk_cache_is_disabled_without_a_directory(tmp_path: Path) -> None:
    """No graph_cache_dir means no key and no writes."""
    dataset = DockQDataset([_entry(tmp_path / "1abc_HA.pdb")])
    assert dataset._disk_cache_path(dataset.entries[0]) is None


def test_feature_version_is_part_of_the_cache_key(tmp_path: Path) -> None:
    """Bumping FEATURE_VERSION must invalidate every cached graph."""
    dataset = DockQDataset([_entry(tmp_path / "1abc_HA.pdb")], graph_cache_dir=str(tmp_path / "c"))
    before = dataset._disk_cache_path(dataset.entries[0])
    import confidock.dockq_predictor.data.dataset as dataset_module

    dataset_module.FEATURE_VERSION = FEATURE_VERSION + 1
    try:
        assert dataset._disk_cache_path(dataset.entries[0]) != before
    finally:
        dataset_module.FEATURE_VERSION = FEATURE_VERSION


def test_filtered_dataset_substitutes_a_decoy_of_the_same_complex(tmp_path: Path) -> None:
    """A failed sample is replaced from its own complex, keeping the ranking chunk whole."""
    sibling = _two_chain_pdb(tmp_path / "1abc_HA_sample_1.pdb")
    other_complex = _two_chain_pdb(tmp_path / "5xyz_HA_sample_0.pdb")
    bad = tmp_path / "1abc_HA_sample_0.cif"
    bad.write_text("not a structure\n")
    dataset = FilteredDockQDataset([_entry(bad), _entry(sibling), _entry(other_complex)], interface_cutoff=6.0, knn_k=4)

    data = dataset[0]  # index 0 is the broken entry
    assert data is not None
    assert data.native_id == "1abc_HA"
    # The one loadable substitute -- never the healthy decoy of the other complex, which
    # the ranking loss would regroup into a singleton and drop.
    assert data.pdb_path == str(sibling)


def test_filtered_dataset_does_not_substitute_across_complexes(tmp_path: Path) -> None:
    """A complex with no usable decoy raises, rather than borrowing from another one."""
    _two_chain_pdb(tmp_path / "1abc_HA_sample_0.pdb")  # healthy, different complex
    bad = tmp_path / "9zzz_HA_sample_0.cif"
    bad.write_text("not a structure\n")
    dataset = FilteredDockQDataset(
        [_entry(bad), _entry(tmp_path / "1abc_HA_sample_0.pdb")], interface_cutoff=6.0, knn_k=4
    )
    with pytest.raises(RuntimeError, match="No usable decoy for complex 9zzz_HA"):
        dataset[0]


def test_filtered_dataset_raises_when_nothing_is_loadable(tmp_path: Path) -> None:
    """All-broken data is a real failure and must not loop forever."""
    bad = tmp_path / "9zzz_HA.cif"
    bad.write_text("not a structure\n")
    with pytest.raises(RuntimeError, match="No usable decoy for complex 9zzz_HA"):
        FilteredDockQDataset([_entry(bad)], knn_k=4)[0]


def test_filtered_dataset_names_the_bad_file_once(tmp_path: Path, capsys) -> None:
    """The substitution is invisible otherwise; one line per file is what allows pruning."""
    sibling = _two_chain_pdb(tmp_path / "1abc_HA_sample_1.pdb")
    bad = tmp_path / "1abc_HA_sample_0.cif"
    bad.write_text("not a structure\n")
    dataset = FilteredDockQDataset([_entry(bad), _entry(sibling)], interface_cutoff=6.0, knn_k=4)

    dataset[0]
    first = capsys.readouterr().out
    assert str(bad) in first
    assert "Substituting another decoy of the same complex" in first

    dataset[0]  # hit once per epoch for the length of a run; must not repeat the warning
    assert "Substituting another decoy" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# create_data_splits
# ---------------------------------------------------------------------------


def _entries_for(pdb_ids, decoys: int = 4) -> list:
    return [
        {"pdb_path": f"/d/{pdb_id}_HLA_sample_{k}.cif", "model_source": "boltz2", "dockq": 0.5}
        for pdb_id in pdb_ids
        for k in range(decoys)
    ]


def test_splits_never_share_a_pdb_id() -> None:
    """A row-wise split would measure memorisation; this must split by pdb_id."""
    entries = _entries_for([f"{i:04d}" for i in range(20)])
    train, val, test = create_data_splits(entries, val_fraction=0.15, test_fraction=0.15, random_seed=7)

    def pdb_ids(split):
        return {Path(e["pdb_path"]).name.split("_")[0] for e in split}

    assert pdb_ids(train) & pdb_ids(val) == set()
    assert pdb_ids(train) & pdb_ids(test) == set()
    assert pdb_ids(val) & pdb_ids(test) == set()
    assert len(train) + len(val) + len(test) == len(entries)


def test_splits_keep_alternative_binding_modes_of_one_pdb_id_together() -> None:
    """Grouping by pdb_id (not native_id) is the stricter choice, so pin it."""
    entries = [
        {"pdb_path": "/d/1a3r_HLP_sample_0.cif", "model_source": "boltz2", "dockq": 0.5},
        {"pdb_path": "/d/1a3r_ABC_sample_0.cif", "model_source": "boltz2", "dockq": 0.5},
    ] + _entries_for([f"{i:04d}" for i in range(9)])
    splits = create_data_splits(entries, random_seed=3)
    for split in splits:
        modes = [e["pdb_path"] for e in split if "1a3r" in e["pdb_path"]]
        assert len(modes) in (0, 2)


def test_splits_are_reproducible_for_a_seed() -> None:
    """Same seed, same split -- replicates must be comparable."""
    entries = _entries_for([f"{i:04d}" for i in range(20)])
    first = create_data_splits(entries, random_seed=11)
    second = create_data_splits(entries, random_seed=11)
    assert [[e["pdb_path"] for e in s] for s in first] == [[e["pdb_path"] for e in s] for s in second]


# ---------------------------------------------------------------------------
# GroupedComplexBatchSampler
# ---------------------------------------------------------------------------


def _sampler_entries(n_complexes: int, decoys: int) -> list:
    return [
        {"pdb_path": f"/d/{i:04d}_HLA_sample_{k}.cif", "model_source": "boltz2", "dockq": 0.5}
        for i in range(n_complexes)
        for k in range(decoys)
    ]


def test_sampler_packs_multiple_decoys_of_the_same_complex_per_batch() -> None:
    """The whole reason it exists: the ranking term needs >=2 same-complex decoys."""
    entries = _sampler_entries(n_complexes=10, decoys=8)
    sampler = GroupedComplexBatchSampler(entries, batch_size=32, decoys_per_complex=8)

    for batch in sampler:
        counts: dict = {}
        for idx in batch:
            counts.setdefault(_native_id_from_path(entries[idx]["pdb_path"]), []).append(idx)
        assert all(len(v) >= 2 for v in counts.values())
        assert len(counts) <= 32 // 8


def test_sampler_covers_every_decoy_exactly_once_per_epoch() -> None:
    """No decoy is dropped or duplicated when the chunks divide evenly."""
    entries = _sampler_entries(n_complexes=6, decoys=8)
    batches = list(GroupedComplexBatchSampler(entries, batch_size=32, decoys_per_complex=8))
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(len(entries)))


def test_sampler_drops_a_size_one_tail_chunk() -> None:
    """A lone decoy carries no ranking signal, so it is not emitted."""
    entries = _sampler_entries(n_complexes=1, decoys=5)  # 5 = 4 + a size-1 tail
    batches = list(GroupedComplexBatchSampler(entries, batch_size=8, decoys_per_complex=4))
    assert sum(len(b) for b in batches) == 4


def test_sampler_forces_at_least_two_decoys_per_complex() -> None:
    """decoys_per_complex=1 would make every chunk a no-op, so it is clamped."""
    sampler = GroupedComplexBatchSampler(_sampler_entries(4, 4), batch_size=8, decoys_per_complex=1)
    assert sampler.decoys_per_complex == 2


def test_sampler_reshuffles_each_epoch() -> None:
    """Batch composition must vary across epochs even though the chunks are fixed."""
    sampler = GroupedComplexBatchSampler(_sampler_entries(12, 8), batch_size=32, decoys_per_complex=8)
    first = list(sampler)
    second = list(sampler)
    assert sampler.epoch == 2
    assert first != second
    assert sorted(i for b in first for i in b) == sorted(i for b in second for i in b)


def test_sampler_length_matches_what_it_yields() -> None:
    """len() is what Lightning uses for the progress bar and the LR schedule."""
    sampler = GroupedComplexBatchSampler(_sampler_entries(7, 8), batch_size=32, decoys_per_complex=8)
    assert len(sampler) == len(list(sampler))


def test_sampler_keeps_at_least_one_complex_per_batch() -> None:
    """A batch_size below decoys_per_complex must not yield empty batches."""
    sampler = GroupedComplexBatchSampler(_sampler_entries(4, 8), batch_size=4, decoys_per_complex=8)
    batches = list(sampler)
    assert batches and all(len(b) == 8 for b in batches)


# ---------------------------------------------------------------------------
# DockQDataModule
# ---------------------------------------------------------------------------


def _datamodule_entries(tmp_path: Path, n_complexes: int = 8, decoys: int = 4) -> list:
    """Real (tiny) structures on disk, one file per decoy."""
    entries = []
    for i in range(n_complexes):
        for k in range(decoys):
            pdb = _two_chain_pdb(tmp_path / f"{i:04d}_HA_sample_{k}.pdb")
            entries.append(_entry(pdb, dockq=0.1 * k))
    return entries


def test_datamodule_setup_splits_and_builds_filtered_datasets(tmp_path: Path) -> None:
    """setup('fit') must leave train/val datasets that skip unloadable samples."""
    dm = DockQDataModule(_datamodule_entries(tmp_path), batch_size=4, num_workers=0)
    dm.setup("fit")

    assert isinstance(dm.train_dataset, FilteredDockQDataset)
    assert isinstance(dm.val_dataset, FilteredDockQDataset)
    assert dm.test_dataset is None  # not requested by the 'fit' stage
    assert len(dm.train_entries) + len(dm.val_entries) + len(dm.test_entries) == 32


def test_datamodule_setup_is_idempotent_about_the_split(tmp_path: Path) -> None:
    """setup() runs on every GPU; it must not reshuffle the split each time."""
    dm = DockQDataModule(_datamodule_entries(tmp_path, n_complexes=4), batch_size=4, num_workers=0)
    dm.setup("fit")
    first = [e["pdb_path"] for e in dm.train_entries]
    dm.setup("fit")
    assert [e["pdb_path"] for e in dm.train_entries] == first


def test_datamodule_passes_the_graph_settings_to_every_split(tmp_path: Path) -> None:
    """A split built with a different cutoff would silently mix feature sets."""
    dm = DockQDataModule(
        _datamodule_entries(tmp_path, n_complexes=4),
        batch_size=4,
        num_workers=0,
        interface_cutoff=6.0,
        knn_k=4,
        cache_graphs=False,
        graph_cache_dir=str(tmp_path / "cache"),
    )
    dm.setup(None)
    for dataset in (dm.train_dataset, dm.val_dataset, dm.test_dataset):
        assert dataset.interface_cutoff == pytest.approx(6.0)
        assert dataset.knn_k == 4
        assert dataset.cache_graphs is False
        assert dataset.graph_cache_dir == str(tmp_path / "cache")


def test_train_dataloader_uses_the_grouped_sampler_when_asked(tmp_path: Path) -> None:
    """group_by_complex swaps batch_size/shuffle for the grouped batch_sampler."""
    dm = DockQDataModule(
        _datamodule_entries(tmp_path, n_complexes=4),
        batch_size=8,
        num_workers=0,
        group_by_complex=True,
        decoys_per_complex=4,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert isinstance(loader.batch_sampler, GroupedComplexBatchSampler)
    assert loader.batch_sampler.decoys_per_complex == 4


def test_train_dataloader_shuffles_when_not_grouping(tmp_path: Path) -> None:
    """The ungrouped path is plain shuffled batching."""
    dm = DockQDataModule(_datamodule_entries(tmp_path, n_complexes=4), batch_size=8, num_workers=0)
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert not isinstance(loader.batch_sampler, GroupedComplexBatchSampler)
    assert loader.batch_size == 8


def test_val_and_test_dataloaders_do_not_shuffle(tmp_path: Path) -> None:
    """Evaluation order must be stable so per-native grouping is reproducible."""
    dm = DockQDataModule(_datamodule_entries(tmp_path, n_complexes=4), batch_size=8, num_workers=0)
    dm.setup(None)
    from torch.utils.data import SequentialSampler

    assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
    assert isinstance(dm.test_dataloader().sampler, SequentialSampler)


@pytest.mark.parametrize("which", ["train", "val", "test"])
def test_dataloaders_require_setup_first(which: str) -> None:
    """A clear error beats an AttributeError on None."""
    dm = DockQDataModule([], num_workers=0)
    with pytest.raises(RuntimeError, match="setup"):
        getattr(dm, f"{which}_dataloader")()


def test_train_dataloader_yields_a_batch_carrying_native_ids(tmp_path: Path) -> None:
    """End-to-end: the grouped loader delivers what training_step needs.

    ``pin_memory`` is off only so the test does not need free device memory --
    pinning is a host-side transfer optimisation, not part of what is asserted.
    """
    dm = DockQDataModule(
        _datamodule_entries(tmp_path, n_complexes=4, decoys=4),
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        interface_cutoff=6.0,
        knn_k=4,
        group_by_complex=True,
        decoys_per_complex=4,
        val_fraction=0.0,
        test_fraction=0.0,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    assert batch.x.shape[1] == NODE_FEATURE_DIM
    assert batch.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert len(batch.native_id) == batch.y_dockq.numel()
    assert max(batch.native_id.count(nid) for nid in set(batch.native_id)) >= 2


def test_loader_worker_init_pins_one_torch_thread() -> None:
    """N workers inheriting the host's 128-thread pool measured ~3x slower."""
    from confidock.dockq_predictor.data.datamodule import _init_loader_worker

    original = torch.get_num_threads()
    try:
        _init_loader_worker(0)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def test_repr_reports_the_configuration(tmp_path: Path) -> None:
    """Run logs quote the repr, so keep it informative."""
    dm = DockQDataModule(_datamodule_entries(tmp_path, n_complexes=2), batch_size=16, num_workers=0)
    text = repr(dm)
    assert "n_entries=8" in text
    assert "batch_size=16" in text
    assert repr(DockQDataset([], knn_k=4)) == "DockQDataset(0 samples)"


def test_graph_cache_dir_is_created_if_missing(tmp_path: Path) -> None:
    """A first run must not fail because the cache directory does not exist yet."""
    cache_dir = tmp_path / "nested" / "cache"
    DockQDataset([], graph_cache_dir=str(cache_dir))
    assert os.path.isdir(cache_dir)
