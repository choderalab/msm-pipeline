import pyemma
import numpy as np

def write_pdbs_of_clusters(source, msm, project_name, n_samples=10, max_states=100):
    '''
    For each state `i` in the connected set of the `msm`, draw `n_samples` frames
    from that state uniformly at random, and write them out to `{project_name}_state_{i}.pdb`.

    If the connected set of the MSM has greater than `max_states` states, only do this for
    the top-`max_states` most populous states.

    Parameters
    ----------
    source : pyemma DataSource object
      trajectory source files

    msm : pyemma MSM object
      MSM estimated from the source files

    project_name : string
      project name, used for naming output files

    n_samples : integer
      number of samples to draw from each state

    max_states : integer
      maximum number of PDB files to write out
    '''
    samples = msm.sample_by_state(n_samples)
    n_states = len(msm.stationary_distribution)

    # only take the top-max_states clusters
    if n_states > max_states:
        indices = sorted(np.arange(n_states), key = lambda i: -msm.stationary_distribution[i])
    else:
        indices = np.arange(n_states)

    for i in indices:
        pyemma.coordinates.save_traj(source, samples[i], '{0}_state_{1}.pdb'.format(project_name, i))

def get_atom_indices(traj, types = ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4']):
    '''
    Fetches the atom indices of all dihedral angles.

    Returns
    -------
    atom_indices : (n_dihedrals, 4)-array
      the 4 atoms involved in each matching dihedral
    '''
    atom_indices = []
    for dih_type in types:
        func = getattr(md, 'compute_{0}'.format(dih_type))
        atoms,_ = func(traj[0])
        atom_indices.append(atoms)
    return np.vstack(atom_indices)

def truncate_unassigned(dtrajs, unassigned_label = -1):
    '''
    Handles an error that sometimes occurs when frames cannot be assigned to
    cluster labels, causing MSM estimation to fail.

    For each dtraj, truncates it at the first instance of an unassigned_label.
    If the first element is invalid, discard the whole discrete trajectory.

    Parameters
    ----------
    dtrajs : list of integer-valued arrays
        Discrete trajectories to analyze, possibly containing invalid state labels
    unassigned_label : integer
        What label is invalid

    Returns
    -------
    safe_dtrajs : list of integer-valued arrays
        Truncates any dtraj containing an invalid label, discards any trajectory
        whose first element is invalid.
        
    discard_pile : list of integers
        Indices of any discarded trajectories.
    '''
    safe_dtrajs = []
    discard_pile = []

    for i,dtraj in enumerate(dtrajs):
        if unassigned_label in set(dtraj):
            bad_index = np.argmax(dtraj == unassigned_label)
            if bad_index > 0:
                safe_dtrajs.append(dtraj[:bad_index])
            else:
                discard_pile.append(i)
                print('Warning! The first element of one of the dtrajs was invalid -- discarding.')
        else:
            safe_dtrajs.append(dtraj)
        return safe_dtrajs, discard_pile
