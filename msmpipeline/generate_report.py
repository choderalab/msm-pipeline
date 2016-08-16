import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyemma
import corner
import numpy as np

def make_plots(dtrajs, tica, tica_output, msm, project_name):
    '''
    Plots diagnostics and sanity-check figures.

    Parameters
    ----------
    dtrajs : list of int-arrays
        discrete trajectories

    tica_output : list of (n_i,d)-arrays
        list of tICA-transformed trajectories

    msm : pyemma MSM object
        estimated Markov State Model

    project_name : string
        name of project, will be used in filenames for figures
    '''

    # make initial plots
    plot_trajectory_length_histogram(dtrajs, project_name)

    # make tICA plots
    inspect_tica(tica, project_name)
    plot_tics(tica_output, n_tics = 10, project_name = project_name)

    # make MSM plots
    plot_sanity_check(msm, project_name)
    compute_its(dtrajs, project_name)
    plot_timescales(msm, project_name)

    # coarse-grain
    n_states = estimate_n_macrostates(msm)
    hmm = msm.coarse_grain(n_states)

    # make HMM plots
    plot_free_energies(hmm, project_name)

    # sample from macrostates
    # to-do!
    # follow: http://www.emma-project.org/latest/generated/MSM_BPTI.html#representative-structures

def inspect_tica(tica, project_name):
    '''
    Plot the cumulative kinetic variance explained by the tICA model (sum of squared tICA eigenvalues).

    Save figure to '{project_name}_tica_kinetic_variance.png'.

    Parameters
    ----------
    tica : pyemma tICA object

    project_name : string
    '''
    # plot cumulative kinetic variance explained
    eigs = tica.eigenvalues
    plt.figure()
    plt.plot(np.cumsum(eigs**2))
    plt.xlabel('# tICA eigenvalues')
    plt.ylabel('Cumulative sum of tICA eigenvalues squared')
    plt.title('Cumulative "kinetic variance" explained')
    plt.savefig('{0}_tica_kinetic_variance.png'.format(project_name),dpi=300)
    plt.close()

def plot_tics(Y, n_tics, project_name):
    '''
    Generate corner plots from tICA projection.

    Save figure to '{project_name}_tica_projection.png'.

    Parameters
    ----------
    Y : list of (n_i,d)-arrays
        list of tICA-transformed trajectories

    n_tics : integer
        number of tICs to plot; the resulting number of subplots will be `n_tics * (n_tics - 1) / 2`

    project_name : string
    '''
    plt.figure()
    Y_ = np.vstack(Y)[:,:n_tics]
    labels = ['tIC{0}'.format(i+1) for i in range(Y_.shape[1])]
    corner.corner(Y_, labels = labels, bins = 50)
    plt.title('Projection onto top-{0} tICs'.format(len(labels)))
    plt.savefig('{0}_tica_projection.png'.format(project_name),dpi=300)
    plt.close()

def plot_sanity_check(msm, project_name):
    ''' Plot stationary distribution vs. counts.

    (The MSM transition matrix induces an estimate of the stationary distribution over microstates.
    How different is this estimate from the raw counts?)


    Saves figure to '{project_name}_sanity_check.png'.

    Parameters
    ----------
    msm : pyemma MSM object

    project_name : string
    '''
    statdist = msm.stationary_distribution
    relative_counts = msm.count_matrix_active.sum(0) / np.sum(msm.count_matrix_active)

    plt.figure()
    plt.scatter(statdist, relative_counts)
    plt.xlabel('MSM stationary distribution')
    plt.ylabel('Relative counts')
    plt.savefig('{0}_sanity_check.png'.format(project_name), dpi = 300)
    plt.close()

def plot_trajectory_length_histogram(dtrajs, project_name):
    '''
    Plots the distribution of trajectory lengths.

    Saves figure to '{project_name}_traj_length_histogram.png'.

    Parameters
    ----------
    dtrajs : list of integer-arrays
        discrete trajectories

    project_name : string
    '''
    lens = []
    for dtraj in dtrajs:
        lens.append(len(dtraj))
    plt.figure()
    plt.hist(lens, bins = 50);
    plt.xlabel('Trajectory length')
    plt.ylabel('Occurrences')
    plt.title('Distribution of trajectory lengths')
    plt.savefig('{0}_traj_length_histogram.png'.format(project_name), dpi = 300)
    plt.close()

def plot_timescales(msm, project_name):
    '''
    Plots the implied timescales from an MSM.

    Saves figure to '{project_name}_timescales.png'.

    Parameters
    ----------
    msm : pyemma MSM object

    project_name : string

    '''
    plt.figure()
    plt.plot(msm.timescales() / 4, '.') # note: hard-coded division by 4 here, corresponding to 250ps between frames
    plt.xlabel('Timescale index')
    plt.ylabel('Timescale (ns)')
    plt.title('{0}: Timescales'.format(project_name))
    plt.savefig('{0}_timescales.png'.format(project_name), dpi = 300)
    plt.close()

def compute_its(dtrajs, project_name):
    '''
    To select lag-time for MSM estimation, we look for the earliest lag-time where these curves flatten out /
    become statistically indistinguishable from flat.

    By default, will produce two figures, one that looks at lags of 1-1000 and one that zooms in on 1-100.

    Saves figures to '{project_name}_its_0.png' and '{project_name}_its_1.png'

    Parameters
    ----------
    dtrajs : list of integer-arrays
        discrete trajectories

    project_name : string

    '''

    lag_sets = [ range(1,101),
                 range(1,1001)[::10]
               ]

    for i,lags in enumerate(lag_sets):
        its = pyemma.msm.its(dtrajs, lags, nits = 20, errors = 'bayes')
        plt.figure()
        pyemma.plots.plot_implied_timescales(its, units = 'ns', dt = 0.25)
        plt.savefig('{0}_its_{1}.png'.format(project_name,i), dpi = 300)
        plt.close()

def estimate_n_macrostates(msm, metastability_threshold=400):
    '''
    Estimate the number of macrostates that are more metastable than some threshold (in units of frames).

    Parameters
    ----------
    msm : pyemma MSM object

    metastability_threshold : integer

    Returns
    -------
    n_macrostates : integer
        number of MSM timescales that exceeded the `metastability_threshold`
    '''
    return sum(msm.timescales() > metastability_threshold)

def plot_free_energies(cg_model, project_name):
    '''
    Given a coarse-grained MSM, plot the relative free energies of each macrostate.

    Saves figure to '{project_name}_macrostate_free_energies.png'.

    Parameters
    ----------
    cg_model : pyemma MSM or HMM object
        coarse-grained model

    project_name : string
    '''
    f_i = -np.log(sorted(cg_model.stationary_distribution))[::-1]
    f_i -= f_i.min()
    plt.figure()
    plt.plot(f_i, '.')
    plt.xlabel('Macrostate')
    plt.ylabel(r'$\Delta G$ $(k_B T)$')
    plt.title('Macrostate free energies')
    plt.savefig('{0}_macrostate_free_energies.png'.format(project_name), dpi = 300)
    plt.close()
