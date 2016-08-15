import mdtraj as md
import cPickle
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyemma
from time import time
from utils import get_atom_indices, write_pdbs_of_clusters

def run_pipeline(fnames,
                 msm_lag = 50,
                 project_name = 'abl',
                 n_clusters = 500,
                 d_threshold = 0.4, # nanometers
                 max_respairs = 1000,
                 max_tics = 500,
                 respair_stride = 100,
                 metastability_threshold = 400, # 100ns (in units of 0.25ns)
                 quick_model = True
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
    t0 = time()
    print('featurizing and tICA-fying!')
    if not quick_model:
        print('Finding respairs_that_changed...')
        # examine a subset of the data to determine which residue pairs are crossing the threshold
        scheme = 'closest'
        threshold = d_threshold
        from contact_features import find_respairs_that_changed
        respairs_that_changed = find_respairs_that_changed(fnames,
                                                           scheme=scheme,
                                           max_respairs = max_respairs)
        atom_indices = get_atom_indices(traj)

    # featurize all frames
    traj = md.load_frame(fnames[0],0)
    top = traj.top

    feat = pyemma.coordinates.featurizer(top)
    feat.add_backbone_torsions(cossin = True)

    if not quick_model:
        feat.add_dihedrals(atom_indices, cossin = True)
        feat.add_residue_mindist(residue_pairs = respairs_that_changed, scheme = scheme)

    n_features = len(feat.describe())
    print('Num features: ', n_features)

    # do featurization + tICA by streaming over size-1000 "chunks"
    source = pyemma.coordinates.source(fnames, features = feat)
    source_full = pyemma.coordinates.source(fnames, top = top)
    if quick_model:
        max_tics = 50
    max_tics = min(max_tics, n_features)
    tica = pyemma.coordinates.tica(lag = 10, kinetic_map = True, dim = max_tics)
    stages = [source, tica]
    pipeline = pyemma.coordinates.pipeline(stages, chunksize = 1000)

    Y = tica.get_output()

    np.save('{0}_tica_eigenvalues.npy'.format(project_name), tica.eigenvalues)
    np.save('{0}_tica_eigenvectors.npy'.format(project_name), tica.eigenvectors)

    print('Done tICA-fying!')
    print('Total elapsed time: ', time() - t0)
    print('Saving tICA projection')
    np.save('{0}_tica_projection.npy'.format(project_name), Y)
    print('Total elapsed time: ', time() - t0)

    print('Discretizing!')
    # discretize
    kmeans = pyemma.coordinates.cluster_mini_batch_kmeans(Y, k = n_clusters, max_iter = 1000)
    dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]

    dtrajs, discard_pile = truncate_unassigned(dtrajs)

    # save outputs
    np.save('{0}_dtrajs.npy'.format(project_name), dtrajs)
    print('Done discretizing! Total elapsed time: ', time() - t0)

    #with open('{0}_tica_model.pkl'.format(project_name),'w') as f:
    #    cPickle.dump(tica, f)

    #with open('{0}_kmeans_model.pkl'.format(project_name),'w') as f:
    #    cPickle.dump(kmeans, f)

    # create and save file index, for later use:

    if len(discard_pile != 0):
        raise Exception("Some dtrajs had to be discarded! File index should be updated.")

    file_index = dict(zip(source.trajfiles, source.trajectory_lengths()))
    with open('{0}_file_index.pkl'.format(project_name), 'w') as f:
        cPickle.dump(file_index, f)

    print('Estimating MSM!')
    msm = pyemma.msm.estimate_markov_model(dtrajs, lag = msm_lag)
    print('Done estimating MSM! Total elapsed time: ', time() - t0)

    print('Writing PDBs!')
    write_pdbs_of_clusters(source_full, msm, project_name)
    print('Done writing PDBs! Total elapsed time: ', time() - t0)
    from generate_report import make_plots
    make_plots(dtrajs, tica, tica_output = Y, msm = msm, project_name = project_name)

if __name__ == '__main__':
    import sys
    path_to_trajs = sys.argv[1]
    project_name = sys.argv[2]

    def get_filenames(path_to_trajs):
        from glob import glob
        filenames = glob(path_to_trajs)
        return filenames

    print(path_to_trajs)
    fnames = get_filenames(path_to_trajs)
    print(fnames)

    print('Running pipeline')
    run_pipeline(fnames, project_name = project_name)
