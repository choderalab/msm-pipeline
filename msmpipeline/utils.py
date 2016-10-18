'''

Partial clone of the VMD plug-ins `cispeptide` and `chirality`

'''

import mdtraj as md
import numpy as np

def find_peptide_bonds(top):
    '''

    Find all peptide bonds in a topology

    Parameters
    ----------
    top : mdtraj topology

    Returns
    -------
    peptide_bonds : list of 4-tuples
        each 4-tuple contains indices of the (CA, N, C, O) atoms involved in the peptide bond

    residues : list of integers
        peptide_bonds[i] involves resids (residues[i], residues[i]+1)

    References
    ----------
    [1] Stereochemical errors and their implications for molecular dynamics simulations.
        Schriener et al., 2011. BMC Bioinformatics. DOI: 10.1186/1471-2105-12-190
        http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-190

    '''

    bond_indices = [sorted((bond[0].index, bond[1].index)) for bond in top.bonds]

    peptide_bonds = []
    residues = []

    for i in range(top.n_residues - 1):

        j = i + 1

        # each of these selections returns an array containing zero or one integer
        O_ind = (top.select('resid=={0} and name == O'.format(i)))
        C_ind = (top.select('resid=={0} and name == C'.format(i)))

        N_ind = (top.select('resid=={0} and name == N'.format(j)))
        CA_ind = (top.select('resid=={0} and name == CA'.format(j)))

        # if C(i) is bonded to N(j), and all of the index arrays are of length-1, add the (CA, N, C, O) tuple
        # to our peptide_bonds list
        if sorted((C_ind, N_ind)) in bond_indices and sum([len(a) != 1 for a in (O_ind, C_ind, N_ind, CA_ind)]) == 0:
            peptide_bonds.append((CA_ind[0], N_ind[0], C_ind[0], O_ind[0]))
            residues.append(i)

    return peptide_bonds, residues

def check_cispeptide_bond(traj, dihedral_cutoff = 85):
    '''
    Given a trajectory, check every peptide bond in every frame.

    Return a boolean array of shape len(traj), n_peptide_bonds

    If any of the peptide bonds appear to be cis-peptide bonds, also print a warning.

    Parameters
    ----------
    traj : mdtraj.Trajectory

    dihed_cutoff : default 85
        Dihedral angle cutoff, in degrees

    Returns
    -------
    problems : boolean array
        problems[i,j] means that, in frame i, peptide bond j is cis

    References
    ----------
    [1] Stereochemical errors and their implications for molecular dynamics simulations.
        Schriener et al., 2011. BMC Bioinformatics. DOI: 10.1186/1471-2105-12-190
        http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-190
    '''

    indices, residues = find_peptide_bonds(traj.top)
    angles = md.compute_dihedrals(traj, indices)
    in_radians = angles * 180 / np.pi
    problems = in_radians > dihedral_cutoff

    if np.sum(problems) > 0:
        print('Problems in {0} frames!'.format(sum(problems.sum(1) != 0)))
        print('Problems in {0} bonds!'.format(sum(problems.sum(0) != 0)))

    return problems


# to-do: check chirality
# approach
# 1. Define collections of chiral atoms for each residue
# 2. Get a list of (id0, id1, id2, id3, (optionally, idH)) tuples
# 3. For each of these tuples,
    # - Compute improper torsion between id0, id1, id2, id3
    # - If there's an H, compute another improper torsion between idH, id1, id2, id3
    # - If either of these impropers is < 0, then it's a problem!
def get_chiral_atoms(residue_name):
    '''

    Returns
    -------

    '''

    raise NotImplementedError()
    #chiral_atoms = []
    #AAs = 'ALA ARG ASN ASP CYS GLN GLU HSP HSD HIP HIE HID HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
    #if residue_name in AAs:
     #   atoms = 'HA CA N C CB'.split()
     #   chiral_atoms.append()

    #return chiral_atoms
