import numpy as np
from utils import check_chirality, check_cispeptide_bond, get_chiral_atoms, find_peptide_bonds

# load "negative control" example trajectory
from msmbuilder.example_datasets import MinimalFsPeptide
traj = MinimalFsPeptide().get().trajectories[0]
top = traj.top

def test_get_chiral_atoms():
    """check that get_chiral_atoms returns a non-empty list, and that the list only
    contains 4-tuples"""
    chiral_atoms = get_chiral_atoms(top)
    print('List of chiral atoms:')
    print(chiral_atoms)
    assert(len(chiral_atoms) > 0)
    assert(set([len(quartet) for quartet in chiral_atoms]) == {4})

def test_find_peptide_bonds():
    """check that find_peptide_bonds returns two lists of the same length """
    peptide_bonds, residues = find_peptide_bonds(top)
    assert(len(peptide_bonds) == len(residues))

def test_check_cispeptide_bond():
    """check that this doesn't erroneously report cispeptide bond errors on error-free structures,
    and that it correctly identifies errors on structures that contain them """

    # negative control
    result = check_cispeptide_bond(traj)
    assert(np.sum(result) == 0)

    # to-do: positive controls

    # to-do: add tests from the VMD plug-in

def test_check_chirality():
    """check whether we erroneously detect chirality errors in a clean trajectory"""
    # negative control
    chiral_atoms, errors = check_chirality(traj)
    print('How many errors: {0}\n\tHow many sites have errors: {1}\n\tHow many frames have errors: {2}'.format(
        np.sum(errors),
        (np.sum(errors, 0) > 0).sum(),
        (np.sum(errors, 1) > 0).sum()))
    assert(np.sum(errors) == 0)

    # to-do: positive controls
    # to-do: load in alanine dipeptide example
    # to-do: load in abl example

# run tests
test_get_chiral_atoms()
test_find_peptide_bonds()
test_check_cispeptide_bond()
test_check_chirality()
