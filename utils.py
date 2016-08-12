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
