'''

Given PDB outputs from this version of `msm-pipeline`, align them all
 to the minimum free energy structure and save them again.

'''

import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md

from glob import glob

def load_structures_and_free_energies(path_to_pdbs, path_to_free_energies):
    '''
    Given paths to PDBs and free energies, return list of trajectories + free energy array

    Parameters
    ----------
    path_to_pdbs : string
        path (containing wildcards) to PDB files output from `msm-pipeline`

    path_to_free_energies: string
        path to .npy file containing macrostate relative free energies

    Returns
    -------
    f_i : numpy.ndarray
        array of macrostate free energies

    all_structures : list of mdtraj.Trajectory objects
        each trajectory contains configurations sampled from a single macrostate
        the list is sorted in ascending order by relative free energy
    '''


    # get filenames of the macrostate PDBs
    fnames = glob(path_to_pdbs)

    # get list of trajectories
    all_structures_unsorted = map(md.load, fnames)

    # load free energies
    f_i = np.load(path_to_free_energies)

    # sort in ascending order by relative free energy
    all_structures = [all_structures_unsorted[i] for i in np.argsort(f_i)]

    return f_i, all_structures


def align_and_save_pdbs(f_i, all_structures, project_name):
    '''

    Aligns all structures to one of the configurations sampled from the minimum
    free energy state, and saves PDBs

    Parameters
    ----------
    f_i : numpy.ndarray
        array of macrostate free energies

    all_structures : list of mdtraj.Trajectory objects
        each trajectory contains configurations sampled from a single macrostate
        the list is sorted in ascending order by relative free energy

    project_name : string
        name of project, used in output filename

    '''

    # reference structure is the first configuration sampled in macrostate 0
    reference = all_structures[0][0]

    # for each collection of macrostate samples
    for i in range(len(all_structures)):
        # align to reference
        aligned = all_structures[i].superpose(reference)

        # filename will look like 'abl_aligned_delta_G_3-04', where we replaced the decimal point with a hyphen
        fname = '{project_name}_aligned_delta_G_{free_energy:.3f}'.format(project_name=project_name,
                                                                          free_energy=sorted(f_i)[i])
        fname = fname.replace('.', '-')

        # save PDB
        aligned.save_pdb(fname + '.pdb')

def compute_and_plot_rmsd_matrix(f_i, all_structures, project_name):
    '''

    Makes a plot of the all-atom RMSD between all macrostates drawn.

    This may help as a sanity check, since we would expect that the samples drawn from a single macrostate
    are "compact" w.r.t. RMSD, which would appear as smaller average values on the diagonal blocks of this matrix.

    Parameters
    ----------
     f_i : numpy.ndarray
        array of macrostate free energies

    all_structures : list of mdtraj.Trajectory objects
        each trajectory contains configurations sampled from a single macrostate
        the list is sorted in ascending order by relative free energy

    project_name : string
        name of project, used in output filename

    Returns
    -------
    rmsd_matrix : numpy.ndarray (square, symmetric)
        matrix containing all-atom RMSD distances between every pair of configurations sampled

    '''
    # concatenate all and superpose
    structures = all_structures[0].join(all_structures[1:])
    structures = structures.superpose(structures)

    # let's generate a pairwise RMSD matrix: are elements of a macrostate similar to each other?
    rmsd_matrix = np.zeros((len(structures), len(structures)))
    for i in range(len(structures)):
        rmsd_matrix[i] = md.rmsd(structures, structures[i])

    # draw blocks around each state
    ticks = np.cumsum(map(len, all_structures)) - 1 # I think there was originally an off-by-one error here, and I think -1 solves it?

    # plot rmsd_matrix
    plt.figure()
    plt.imshow(rmsd_matrix, interpolation='none', cmap='Blues')
    plt.xticks(ticks, [''] * len(ticks))
    plt.yticks(ticks, [''] * len(ticks))
    plt.grid(True, linestyle='solid')
    plt.colorbar()
    plt.title('Pairwise RMSD\n({0} macrostate samples)'.format(project_name))
    fname = '{project_name}_macrostate_rmsd_matrix.png'.format(project_name=project_name)
    plt.savefig(fname)
    plt.close()

    return rmsd_matrix

def main():
    # parse paths and project_name from command line
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-t", "--path_to_pdbs", dest="path_to_pdbs", type="string",
                      help="path to PDBs (must be quoted if wildcards are used)")
    parser.add_option("-n", "--path_to_free_energies", dest="path_to_free_energies", type="string",
                      help="path to free energies")
    parser.add_option("-n", "--name", dest="project_name", type="string",
                      help="project name (used in figure filenames)", default="abl")
    (options, args) = parser.parse_args()

    f_i, all_structures = load_structures_and_free_energies(path_to_pdbs=options.path_to_pdbs,
                                                            path_to_free_energies=options.path_to_free_energies)
    align_and_save_pdbs(f_i=f_i, all_structures=all_structures, project_name=options.project_name)
    rmsd_matrix = compute_and_plot_rmsd_matrix(f_i=f_i, all_structures=all_structures, project_name=options.project_name)

if __name__ == '__main__':
    main()