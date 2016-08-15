import mdtraj as md
import cPickle
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyemma

def run_pipeline(fnames,
                 msm_lag = 50,
                 project_name = 'abl',
                 n_clusters = 500,
                 d_threshold = 0.4, # nanometers
                 max_respairs = 1000,
                 max_tics = 500,
                 respair_stride = 100,
                 metastability_threshold = 400, # 100ns (in units of 0.25ns)
                 ):
    '''
    Generates an MSM using sensible defaults. Computes distance-based and angle-based features, turns
    those features into a kinetic distance metric (using tICA), and then clusters w.r.t. that metric.

    Then produces a series of plots about the results, saving them to `{project_name}_*.jpg`.

    Parameters
    ----------
    fnames : list of strings
      list of paths to trajectories

    msm_lag : integer
      lag-time for MSM estimation, in frames

    project_name : string
      project name, to use in figure filenames

    n_clusters : integer
      number of microstates to use when clustering

    d_threshold : float
      distance threshold-- only residue pairs that cross this
      threshold at least once are considered interesting

    max_respairs : integer
      maximum number of residue pairs to use when featurizing

    max_tics : integer
      maximum number of tICA components to retain

    respair_stride : integer
      what thinning fraction to use when performing heuristic
      feature-selection over all possible residue pairs

    metastability_threshold : integer
      threshold (in frames) for the metastability of a macrostate--
      used to coarse-grain the resulting MSM

    '''
    print('Finding respairs_that_changed...')
    # examine a subset of the data to determine which residue pairs are crossing the threshold
    scheme = 'closest'
    threshold = d_threshold
    from contact_features import find_respairs_that_changed
    respairs_that_changed = find_respairs_that_changed(fnames,
                                                       scheme=scheme,
                                                       threshold=threshold,
                                                       stride = respair_stride,
                                                       max_respairs = max_respairs)

    # featurize all frames using these residue pairs + dihedrals + backbone torsions
    traj = md.load_frame(fnames[0],0)
    top = traj.top

    feat = pyemma.coordinates.featurizer(top)

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

    atom_indices = get_atom_indices(traj)
    feat.add_dihedrals(atom_indices, cossin = True)
    feat.add_residue_mindist(residue_pairs = respairs_that_changed, scheme = scheme)
    feat.add_backbone_torsions(cossin = True)

    # do featurization + tICA by streaming over size-1000 "chunks"
    source = pyemma.coordinates.source(fnames, features = feat)
    tica = pyemma.coordinates.tica(lag = 10, kinetic_map = True, var_cutoff = 0.95)
    stages = [source, tica]
    pipeline = pyemma.coordinates.pipeline(stages, chunksize = 1000)

    Y = tica.get_output()

    # truncate to max_tics
    if Y[0].shape[1] > max_tics:
         Y = [y[:,:max_tics] for y in Y]

    # discretize
    kmeans = pyemma.coordinates.cluster_mini_batch_kmeans(Y, k = n_clusters, max_iter = 1000)
    dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]

    # save outputs
    np.save('{0}_dtrajs.npy'.format(project_name), dtrajs)
    np.save('{0}_tica_projection.npy'.format(project_name), Y)

    #with open('{0}_tica_model.pkl'.format(project_name),'w') as f:
    #    cPickle.dump(tica, f)

    #with open('{0}_kmeans_model.pkl'.format(project_name),'w') as f:
    #    cPickle.dump(kmeans, f)

    # create and save file index, for later use:
    file_index = dict(zip(source.trajfiles, source.trajectory_lengths()))
    with open('{0}_file_index.pkl'.format(project_name), 'w') as f:
        cPickle.dump(file_index, f)

    source_full = pyemma.coordinates.source(fnames, top = top)
    msm = pyemma.msm.estimate_markov_model(dtrajs, lag = msm_lag)
    write_pdbs_of_clusters(source_full, msm, project_name)

    from generate_report import make_plots
    make_plots(dtrajs, tica, tica_output = Y, msm = msm, project_name = project_name)

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

def main():
    import sys
    path_to_trajs = sys.argv[1]
    if len(sys.argv) > 2:
        project_name = sys.argv[2]
    else:
        project_name = 'abl'

    def get_filenames(path_to_trajs):
        from glob import glob
        filenames = glob(path_to_trajs)
        return filenames

    print(path_to_trajs)
    fnames = get_filenames(path_to_trajs)
    print(fnames)

    print('Running pipeline')
    run_pipeline(fnames, project_name = project_name)

if __name__ == '__main__':
    main()
