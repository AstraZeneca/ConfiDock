"""Tests for confidock.evaluation.structures module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from confidock.evaluation.structures import (
    BACKBONE_ATOMS,
    CONTACT_CUTOFF,
    backbone_ca_atoms,
    chain_ids_in_order,
    chain_local_ca_rmsd,
    collect_common_ca_atom_pairs,
    common_residue_keys,
    fit_superposition,
    heavy_atoms,
    residue_contacts,
    residue_key,
    rmsd_from_pairs,
    standard_residues,
    transform_coord,
)


@pytest.fixture
def mock_atom():
    """Create a mock BioPython Atom."""
    atom = Mock()
    atom.element = "C"
    atom.get_name.return_value = "CA"
    atom.coord = np.array([1.0, 2.0, 3.0])
    return atom


@pytest.fixture
def mock_residue():
    """Create a mock BioPython Residue."""
    residue = Mock()
    residue.id = (" ", 42, " ")  # (hetflag, seqnum, icode)

    # Mock CA atom
    ca_atom = Mock()
    ca_atom.element = "C"
    ca_atom.get_name.return_value = "CA"
    ca_atom.coord = np.array([1.0, 2.0, 3.0])

    # Mock sidechain atom
    cb_atom = Mock()
    cb_atom.element = "C"
    cb_atom.get_name.return_value = "CB"
    cb_atom.coord = np.array([1.5, 2.5, 3.5])

    # Mock backbone atoms
    n_atom = Mock()
    n_atom.element = "N"
    n_atom.get_name.return_value = "N"

    residue.__contains__ = lambda self, name: name == "CA"
    residue.__getitem__ = lambda self, name: ca_atom if name == "CA" else None
    residue.__iter__ = lambda self: iter([ca_atom, n_atom, cb_atom])

    return residue


@pytest.fixture
def mock_chain():
    """Create a mock BioPython Chain."""
    chain = Mock()
    chain.id = "A"

    # Create mock residues
    res1 = Mock()
    res1.id = (" ", 1, " ")
    res1.__contains__ = lambda self, name: name == "CA"
    res1.__getitem__ = lambda self, name: Mock(coord=np.array([1.0, 0.0, 0.0]))

    res2 = Mock()
    res2.id = (" ", 2, " ")
    res2.__contains__ = lambda self, name: name == "CA"
    res2.__getitem__ = lambda self, name: Mock(coord=np.array([2.0, 0.0, 0.0]))

    chain.__iter__ = lambda self: iter([res1, res2])
    chain.get_chains = lambda: iter([chain])

    return chain


@pytest.fixture
def mock_structure():
    """Create a mock BioPython Structure."""
    structure = Mock()
    chain_a = Mock()
    chain_a.id = "A"
    chain_b = Mock()
    chain_b.id = "B"

    structure.get_chains.return_value = iter([chain_a, chain_b])
    return structure


class TestChainIdsInOrder:
    """Tests for chain_ids_in_order function."""

    def test_chain_ids_in_order_basic(self, mock_structure):
        """Test extracting chain IDs in order."""
        result = chain_ids_in_order(mock_structure)
        assert result == ["A", "B"]

    def test_chain_ids_in_order_single_chain(self):
        """Test single chain."""
        structure = Mock()
        chain = Mock()
        chain.id = "X"
        structure.get_chains.return_value = iter([chain])

        result = chain_ids_in_order(structure)
        assert result == ["X"]

    def test_chain_ids_in_order_empty(self):
        """Test structure with no chains."""
        structure = Mock()
        structure.get_chains.return_value = iter([])

        result = chain_ids_in_order(structure)
        assert result == []


class TestStandardResidues:
    """Tests for standard_residues function."""

    def test_standard_residues_filters_hetatm(self):
        """Test filtering HETATM residues."""
        chain = Mock()

        std_res = Mock()
        std_res.id = (" ", 1, " ")  # Standard residue (hetflag = " ")

        hetatm_res = Mock()
        hetatm_res.id = ("H_HOH", 2, " ")  # HETATM residue

        chain.__iter__ = lambda self: iter([std_res, hetatm_res])

        result = standard_residues(chain)
        assert len(result) == 1
        assert result[0] == std_res

    def test_standard_residues_empty_chain(self):
        """Test empty chain."""
        chain = Mock()
        chain.__iter__ = lambda self: iter([])

        result = standard_residues(chain)
        assert len(result) == 0


class TestResidueKey:
    """Tests for residue_key function."""

    def test_residue_key_basic(self):
        """Test basic residue key extraction."""
        residue = Mock()
        residue.id = (" ", 42, " ")

        result = residue_key(residue)
        assert result == (42, "")

    def test_residue_key_with_insertion_code(self):
        """Test residue with insertion code."""
        residue = Mock()
        residue.id = (" ", 42, "A")

        result = residue_key(residue)
        assert result == (42, "A")

    def test_residue_key_strips_whitespace(self):
        """Test that whitespace in insertion code is stripped."""
        residue = Mock()
        residue.id = (" ", 42, "  ")

        result = residue_key(residue)
        assert result == (42, "")


class TestHeavyAtoms:
    """Tests for heavy_atoms function."""

    def test_heavy_atoms_excludes_backbone(self):
        """Test that backbone atoms are excluded."""
        residue = Mock()

        # Create atoms
        ca_atom = Mock()
        ca_atom.element = "C"
        ca_atom.get_name.return_value = "CA"

        n_atom = Mock()
        n_atom.element = "N"
        n_atom.get_name.return_value = "N"

        cb_atom = Mock()
        cb_atom.element = "C"
        cb_atom.get_name.return_value = "CB"

        residue.__iter__ = lambda self: iter([ca_atom, n_atom, cb_atom])

        result = heavy_atoms(residue)
        assert len(result) == 1
        assert result[0] == cb_atom

    def test_heavy_atoms_excludes_hydrogens(self):
        """Test that hydrogen atoms are excluded."""
        residue = Mock()

        c_atom = Mock()
        c_atom.element = "C"
        c_atom.get_name.return_value = "CB"

        h_atom = Mock()
        h_atom.element = "H"
        h_atom.get_name.return_value = "HB"

        residue.__iter__ = lambda self: iter([c_atom, h_atom])

        result = heavy_atoms(residue)
        assert len(result) == 1
        assert result[0] == c_atom


class TestBackboneCaAtoms:
    """Tests for backbone_ca_atoms function."""

    def test_backbone_ca_atoms_basic(self, mock_chain):
        """Test extracting CA atoms from chain."""
        result = backbone_ca_atoms(mock_chain)
        assert len(result) == 2
        assert (1, "") in result
        assert (2, "") in result

    def test_backbone_ca_atoms_with_filter(self):
        """Test filtering by residue keys."""
        chain = Mock()

        res1 = Mock()
        res1.id = (" ", 1, " ")
        res1.__contains__ = lambda self, name: name == "CA"
        res1.__getitem__ = lambda self, name: Mock(coord=np.array([1.0, 0.0, 0.0]))

        res2 = Mock()
        res2.id = (" ", 2, " ")
        res2.__contains__ = lambda self, name: name == "CA"
        res2.__getitem__ = lambda self, name: Mock(coord=np.array([2.0, 0.0, 0.0]))

        chain.__iter__ = lambda self: iter([res1, res2])

        # Filter to only include residue 1
        result = backbone_ca_atoms(chain, residue_keys={(1, "")})
        assert len(result) == 1
        assert (1, "") in result
        assert (2, "") not in result


class TestCommonResidueKeys:
    """Tests for common_residue_keys function."""

    def test_common_residue_keys_basic(self):
        """Test finding common residue keys."""
        chain_a = Mock()
        res_a1 = Mock()
        res_a1.id = (" ", 1, " ")
        res_a2 = Mock()
        res_a2.id = (" ", 2, " ")
        chain_a.__iter__ = lambda self: iter([res_a1, res_a2])

        chain_b = Mock()
        res_b1 = Mock()
        res_b1.id = (" ", 1, " ")
        res_b3 = Mock()
        res_b3.id = (" ", 3, " ")
        chain_b.__iter__ = lambda self: iter([res_b1, res_b3])

        result = common_residue_keys(chain_a, chain_b)
        assert result == {(1, "")}

    def test_common_residue_keys_no_overlap(self):
        """Test with no common residues."""
        chain_a = Mock()
        res_a1 = Mock()
        res_a1.id = (" ", 1, " ")
        chain_a.__iter__ = lambda self: iter([res_a1])

        chain_b = Mock()
        res_b2 = Mock()
        res_b2.id = (" ", 2, " ")
        chain_b.__iter__ = lambda self: iter([res_b2])

        result = common_residue_keys(chain_a, chain_b)
        assert result == set()


class TestTransformCoord:
    """Tests for transform_coord function."""

    def test_transform_coord_identity(self):
        """Test with identity transformation."""
        coord = np.array([1.0, 2.0, 3.0])
        rot = np.eye(3)
        tran = np.zeros(3)

        result = transform_coord(coord, rot, tran)
        np.testing.assert_array_almost_equal(result, coord)

    def test_transform_coord_translation_only(self):
        """Test translation only."""
        coord = np.array([1.0, 2.0, 3.0])
        rot = np.eye(3)
        tran = np.array([1.0, 1.0, 1.0])

        result = transform_coord(coord, rot, tran)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_coord_rotation_only(self):
        """Test rotation only (90° around z-axis)."""
        coord = np.array([1.0, 0.0, 0.0])
        # 90° counter-clockwise rotation around z-axis
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        tran = np.zeros(3)

        result = transform_coord(coord, rot, tran)
        # [1,0,0] rotated 90° CCW around z-axis gives [0,-1,0]
        expected = np.array([0.0, -1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestRmsdFromPairs:
    """Tests for rmsd_from_pairs function."""

    def test_rmsd_from_pairs_identical(self):
        """Test RMSD of identical coordinates."""
        coords1 = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
        coords2 = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]

        result = rmsd_from_pairs(coords1, coords2)
        assert result == pytest.approx(0.0)

    def test_rmsd_from_pairs_different(self):
        """Test RMSD of different coordinates."""
        coords1 = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        coords2 = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]

        result = rmsd_from_pairs(coords1, coords2)
        assert result == pytest.approx(1.0)

    def test_rmsd_from_pairs_empty(self):
        """Test with empty coordinate lists."""
        result = rmsd_from_pairs([], [])
        assert result == ""

    def test_rmsd_from_pairs_mismatched_length(self):
        """Test with mismatched lengths."""
        coords1 = [np.array([0.0, 0.0, 0.0])]
        coords2 = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])]

        result = rmsd_from_pairs(coords1, coords2)
        assert result == ""


class TestFitSuperposition:
    """Tests for fit_superposition function."""

    @patch("confidock.evaluation.structures.Superimposer")
    def test_fit_superposition_basic(self, mock_superimposer_class):
        """Test basic superposition fitting."""
        # Setup mock
        mock_si = Mock()
        mock_si.rotran = (np.eye(3), np.zeros(3))
        mock_si.rms = 0.5
        mock_superimposer_class.return_value = mock_si

        native_atoms = [Mock(), Mock()]
        model_atoms = [Mock(), Mock()]

        result = fit_superposition(native_atoms, model_atoms)

        assert result is not None
        rot, tran, rms = result
        np.testing.assert_array_equal(rot, np.eye(3))
        np.testing.assert_array_equal(tran, np.zeros(3))
        assert rms == pytest.approx(0.5)

    def test_fit_superposition_empty_atoms(self):
        """Test with empty atom lists."""
        result = fit_superposition([], [])
        assert result is None

    def test_fit_superposition_mismatched_length(self):
        """Test with mismatched atom list lengths."""
        native_atoms = [Mock()]
        model_atoms = [Mock(), Mock()]

        result = fit_superposition(native_atoms, model_atoms)
        assert result is None


class TestCollectCommonCaAtomPairs:
    """Tests for collect_common_ca_atom_pairs function."""

    def test_collect_common_ca_atom_pairs_basic(self):
        """Test collecting CA atom pairs."""
        # Mock chains with common residues
        native_chain = Mock()
        model_chain = Mock()

        # Native residues
        native_res1 = Mock()
        native_res1.id = (" ", 1, " ")
        native_res1.__contains__ = lambda self, name: name == "CA"
        native_ca1 = Mock()
        native_ca1.coord = np.array([1.0, 0.0, 0.0])
        native_res1.__getitem__ = lambda self, name: native_ca1 if name == "CA" else None

        native_res2 = Mock()
        native_res2.id = (" ", 2, " ")
        native_res2.__contains__ = lambda self, name: name == "CA"
        native_ca2 = Mock()
        native_ca2.coord = np.array([2.0, 0.0, 0.0])
        native_res2.__getitem__ = lambda self, name: native_ca2 if name == "CA" else None

        native_chain.__iter__ = lambda self: iter([native_res1, native_res2])

        # Model residues (same keys)
        model_res1 = Mock()
        model_res1.id = (" ", 1, " ")
        model_res1.__contains__ = lambda self, name: name == "CA"
        model_ca1 = Mock()
        model_ca1.coord = np.array([1.1, 0.1, 0.1])
        model_res1.__getitem__ = lambda self, name: model_ca1 if name == "CA" else None

        model_res2 = Mock()
        model_res2.id = (" ", 2, " ")
        model_res2.__contains__ = lambda self, name: name == "CA"
        model_ca2 = Mock()
        model_ca2.coord = np.array([2.1, 0.1, 0.1])
        model_res2.__getitem__ = lambda self, name: model_ca2 if name == "CA" else None

        model_chain.__iter__ = lambda self: iter([model_res1, model_res2])

        native_atoms, model_atoms = collect_common_ca_atom_pairs(native_chain, model_chain)

        assert len(native_atoms) == 2
        assert len(model_atoms) == 2


class TestChainLocalCaRmsd:
    """Tests for chain_local_ca_rmsd function."""

    @patch("confidock.evaluation.structures.fit_superposition")
    @patch("confidock.evaluation.structures.collect_common_ca_atom_pairs")
    def test_chain_local_ca_rmsd_basic(self, mock_collect, mock_fit):
        """Test CA RMSD calculation."""
        # Mock atom pairs
        mock_collect.return_value = ([Mock(), Mock()], [Mock(), Mock()])

        # Mock superposition
        mock_fit.return_value = (np.eye(3), np.zeros(3), 1.5)

        result = chain_local_ca_rmsd(Mock(), Mock())
        assert result == pytest.approx(1.5)

    @patch("confidock.evaluation.structures.fit_superposition")
    @patch("confidock.evaluation.structures.collect_common_ca_atom_pairs")
    def test_chain_local_ca_rmsd_no_fit(self, mock_collect, mock_fit):
        """Test when superposition fails."""
        mock_collect.return_value = ([], [])
        mock_fit.return_value = None

        result = chain_local_ca_rmsd(Mock(), Mock())
        assert result == ""


class TestResidueContacts:
    """Tests for residue_contacts function."""

    @patch("confidock.evaluation.structures.NeighborSearch")
    def test_residue_contacts_basic(self, mock_ns_class):
        """Test finding residue contacts."""
        # Mock NeighborSearch
        mock_ns = Mock()
        mock_ns_class.return_value = mock_ns

        # Mock chain A
        chain_a = Mock()
        res_a1 = Mock()
        res_a1.id = (" ", 1, " ")
        atom_a1 = Mock()
        atom_a1.element = "C"
        atom_a1.get_name.return_value = "CB"
        atom_a1.coord = np.array([0.0, 0.0, 0.0])
        res_a1.__iter__ = lambda self: iter([atom_a1])
        chain_a.__iter__ = lambda self: iter([res_a1])

        # Mock chain B
        chain_b = Mock()
        res_b1 = Mock()
        res_b1.id = (" ", 10, " ")
        atom_b1 = Mock()
        atom_b1.element = "C"
        atom_b1.get_name.return_value = "CB"
        atom_b1.coord = np.array([3.0, 0.0, 0.0])  # Within 5Å
        res_b1.__iter__ = lambda self: iter([atom_b1])
        chain_b.__iter__ = lambda self: iter([res_b1])

        # Mock NeighborSearch to return res_b1
        mock_ns.search.return_value = [res_b1]

        contacts, residues_a, residues_b = residue_contacts(chain_a, chain_b, cutoff=5.0)

        assert ((1, ""), (10, "")) in contacts
        assert (1, "") in residues_a
        assert (10, "") in residues_b

    def test_residue_contacts_empty_chain_b(self):
        """Test with empty chain B."""
        chain_a = Mock()
        res_a1 = Mock()
        res_a1.id = (" ", 1, " ")
        atom_a1 = Mock()
        atom_a1.element = "C"
        atom_a1.get_name.return_value = "CB"
        res_a1.__iter__ = lambda self: iter([atom_a1])
        chain_a.__iter__ = lambda self: iter([res_a1])

        chain_b = Mock()
        chain_b.__iter__ = lambda self: iter([])

        contacts, residues_a, residues_b = residue_contacts(chain_a, chain_b)

        assert contacts == set()
        assert residues_a == set()
        assert residues_b == set()


class TestConstants:
    """Test module constants."""

    def test_backbone_atoms_constant(self):
        """Test BACKBONE_ATOMS constant."""
        assert BACKBONE_ATOMS == {"N", "CA", "C", "O"}

    def test_contact_cutoff_constant(self):
        """Test CONTACT_CUTOFF constant."""
        assert CONTACT_CUTOFF == pytest.approx(5.0)
