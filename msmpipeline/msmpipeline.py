import mdtraj as md
import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import pyemma

def run_pipeline(fnames,
                 msm_lag = 50,
                 project_name = 'abl',
                 n_clusters = 1000,
                 max_tics = 50,
                 metastability_threshold = 100, # in units of nanoseconds
                 n_structures_per_macrostate = 10,
                 in_memory = True,
                 max_n_macrostates = 20,
                 ):
    '''
    Generates an MSM using sensible defaults. Computes angle-based features, turns
    those features into a kinetic distance metric (using tICA), and then clusters w.r.t. that metric.

    Then produces a series of plots about the results, saving them to `{project_name}_*.png`.

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

    max_tics : integer
      maximum number of tICA components to retain

    metastability_threshold : integer
      threshold (in nanoseconds) for the metastability of a macrostate--
      used to coarse-grain the resulting MSM

    n_structures_per_macrostate : integer
      how many configurations to write to PDB per macrostate

    in_memory : bool
      whether to featurize in one go or to iterate over chunks
      
    max_n_macrostates : int
      override estimated n_macrostates if it exceeds max_n_macrostates
    '''
    ## PARAMETERIZE MSM
    # get first traj + topology
    traj = md.load(fnames[0])
    top = traj.top
    
    # get timestep-- stored in units of picoseconds, converted to units of nanoseconds
    timestep = traj.timestep / 1000
    
    # create featurizer
    feat = pyemma.coordinates.featurizer(top)
    feat.add_backbone_torsions(cossin = True)
    n_features = len(feat.describe())

    dim = min(n_features, max_tics)

    torsions = feat.active_features[0].angle_indexes

    # do featurization + tICA
    source = pyemma.coordinates.source(fnames, features = feat)
    source_full = pyemma.coordinates.source(fnames, top=top)

    if in_memory:
        X = source.get_output()
        tica = pyemma.coordinates.tica(X, lag=msm_lag, kinetic_map=True, dim=dim)

    else:
        tica = pyemma.coordinates.tica(lag=msm_lag, kinetic_map=True, dim=dim)
        stages = [source, tica]
        pipeline = pyemma.coordinates.pipeline(stages, chunksize = 1000)

    Y = tica.get_output()

    # discretize
    kmeans = pyemma.coordinates.cluster_mini_batch_kmeans(Y, k = n_clusters, max_iter = 1000)
    dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]

    # save outputs
    np.save('{0}_dtrajs.npy'.format(project_name), dtrajs)
    np.save('{0}_tica_projection.npy'.format(project_name), Y)

    # estimate msm
    msm = pyemma.msm.estimate_markov_model(dtrajs, lag = msm_lag, dt_traj = '{0} ns'.format(timestep))
    np.save('{0}_transmat.npy'.format(project_name), msm.P)
    print('Trace of transition matrix: {0:.3f}'.format(np.trace(msm.P)))
    print('Active count fraction: {0:.3f}'.format(msm.active_count_fraction))
    print('Active state fraction: {0:.3f}'.format(msm.active_state_fraction))

    ## IDENTIFY AND SAMPLE FROM MACROSTATES

    # estimate n_macrostates
    n_macrostates = sum(msm.timescales() > metastability_threshold)

    # safety checks on this estimate
    print('Estimated n_macrostates: {0}'.format(n_macrostates))
    if n_macrostates < 2:
        print("Yikes! Only estimated < 2 macrostates more metastable than threshold.")
        n_macrostates = 2
    elif n_macrostates > msm.nstates:
        print("Huh? Somehow the MSM had more timescales than states.")
        n_macrostates = msm.nstates
   
    # ignore this estimate if it exceeds max_n_macrostates
    if n_macrostates > max_n_macrostates:
        print("Estimated n_macrostates exceeds max_n_macrostates, reverting to {0}".format(max_n_macrostates))
        n_macrostates = max_n_macrostates
        
    # coarse-grain
    hmm = pyemma.msm.estimate_hidden_markov_model(dtrajs, n_macrostates, msm_lag, maxit=1)

    # get indices
    indices = hmm.sample_by_observation_probabilities(n_structures_per_macrostate)

    # write PDBs
    pyemma.coordinates.save_trajs(source, indices, prefix=project_name, fmt = 'pdb')
    
    # write macrostate free energies
    f_i = -np.log(hmm.stationary_distribution)
    f_i -= f_i.min()
    np.save('{0}_macrostate_free_energies.npy'.format(project_name), f_i)

    ## PLOT DIAGNOSTICS

    # tica eigenvalues
    eigs = tica.eigenvalues
    plt.figure()
    plt.plot(np.cumsum(eigs ** 2))
    plt.xlabel('# tICA eigenvalues')
    plt.ylabel('Cumulative sum of tICA eigenvalues squared')
    plt.title('Cumulative "kinetic variance" explained')
    plt.savefig('{0}_tica_kinetic_variance.png'.format(project_name), dpi=300)
    plt.close()

    # macrostate free energies
    f_i = -np.log(sorted(hmm.stationary_distribution))[::-1]
    f_i -= f_i.min()
    plt.figure()
    plt.plot(f_i, '.')
    plt.xlabel('Macrostate')
    plt.ylabel(r'$\Delta G$ $(k_B T)$')
    plt.title('Macrostate free energies')
    plt.savefig('{0}_macrostate_free_energies.png'.format(project_name), dpi=300)
    plt.close()

    # implied timescales
    lag_sets = [range(1, 101),
                range(1, 1001)[::10]
                ]

    for i, lags in enumerate(lag_sets):
        its = pyemma.msm.its(dtrajs, lags, nits=20, errors='bayes')
        plt.figure()        
        pyemma.plots.plot_implied_timescales(its, units='ns', dt=timestep)
        plt.savefig('{0}_its_{1}.png'.format(project_name, i), dpi=300)
        plt.close()

    # sanity check
    statdist = msm.stationary_distribution
    relative_counts = msm.count_matrix_active.sum(0) / np.sum(msm.count_matrix_active)

    plt.figure()
    plt.scatter(statdist, relative_counts)
    plt.xlabel('MSM stationary distribution')
    plt.ylabel('Relative counts')
    plt.savefig('{0}_sanity_check.png'.format(project_name), dpi=300)
    plt.close()

def main():
    import sys
    path_to_trajs = sys.argv[1]
    project_name = 'abl'
    n_clusters = 1000
    if len(sys.argv) > 2:
        project_name = sys.argv[2]
    if len(sys.argv) > 3:
        n_clusters = int(sys.argv[3])

    def get_filenames(path_to_trajs):
        from glob import glob
        filenames = glob(path_to_trajs)
        return filenames

    print(path_to_trajs)
    fnames = get_filenames(path_to_trajs)
    print(fnames)

    print('Running pipeline')
    run_pipeline(fnames, project_name = project_name, n_clusters = n_clusters)

if __name__ == '__main__':
    main()
