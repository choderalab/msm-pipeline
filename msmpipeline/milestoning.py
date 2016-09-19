import pyemma                    # msms
import numpy as np               # ndarrays
from glob import glob            # files
from tqdm import tqdm            # progress bars
import networkx as nx            # graph algorithms
import matplotlib.pyplot as plt  # plotting
from functools import partial    # currying
from multiprocessing import Pool # parallel

# compute gamma matrix

# version 1
def estimate_gamma_version_1(dtrajs):
    '''
    iniitial implementation of estimator for Gamma
    '''

    dtraj_stack = np.hstack(dtrajs)
    n_states = len(set(dtraj_stack))

    gamma = np.zeros((n_states, n_states))

    for dtraj in tqdm(dtrajs):
        for i in range(n_states):
            indices = list(np.arange(len(dtraj))[dtraj == i]) + [len(dtraj)]
            for t in range(len(indices) - 1):
                js = set(dtraj[indices[t]:indices[t + 1]])
                for j in (js - set([i])):
                    gamma[i, j] += 1

    for i in range(n_states):
        gamma[i] /= np.sum(dtraj_stack == i)

    return gamma

# version 2
def find_subsequences(dtraj, i, include_end=True):
    '''
    Splits the dtraj into subsequences that
    1. start with i,
    2. hit at least one state j =/= i before returning to i

    If include_end == True, then we also include

    If include_end == False, we only keep subsequences that have returned to i
        Not implmented yet!

        In the paper, I think they always include_end.


    Returns
    -------
    list of integer arrays

    '''

    if include_end == False:
        raise (NotImplementedError())

    inds = list(np.arange(len(dtraj))[dtraj == i])
    inds.append(len(dtraj))

    subsequences = [dtraj[inds[t]:inds[t + 1]] for t in range(len(inds) - 1)]
    subsequences = [subseq for subseq in subsequences if len(subseq) > 1]
    return subsequences

def estimate_gamma_version_2(dtrajs):
    '''


    Estimates a matrix Gamma, where Gamma[i,j] is the probability that, after leaving `i`,
    you will hit `j` before hitting `i` again.

    This is the "reference" implementation, implemented inefficiently but in a way that's (hopefully) easy to check for correctness.

    Parameters
    ----------
    dtrajs : list of integer arrays
        discrete trajectories, e.g. after assigning to microstates, or trial milestones

    Returns
    -------
    gamma : numpy.ndarray
        square matrix of floats between 0 and 1
        (should this be row-stochastic?)

    '''

    # states are non-negative integers
    assert (np.hstack(dtrajs).min() >= 0)
    n_states = np.max(np.hstack(dtrajs)) + 1

    # matrix we'll store the results
    gamma = np.zeros((n_states, n_states))

    # number of pieces that start in i and hit j
    N_ij = np.zeros((n_states, n_states))

    # number of pieces that start in i
    N_i = np.zeros(n_states)

    # iterate over all dtrajs
    for dtraj in tqdm(dtrajs):
        for i in range(n_states):
            # find subsequences that start in i and hit something else before returning to i
            subsequences = find_subsequences(dtraj, i)

            # increment the count of number of pieces that start in i
            N_i[i] += len(subsequences)

            # for each subsequence, get the set of states visited after i
            sets = [set(subseq[1:]) for subseq in subsequences]

            for s in sets:
                for j in s:
                    N_ij[i, j] += 1

    # estimate gamma from N_ij and N_i
    for i in range(n_states):
        for j in range(n_states):
            gamma[i, j] = N_ij[i, j] / N_i[i]

    return gamma

def get_diversities(dtrajs):
    '''
    how many different microstates are visited in each trajectory?
    '''
    diversities = [len(set(d)) for d in dtrajs]
    plt.hist(diversities, bins=50);
    plt.xlabel('Number of distinct microstates visited in a trajectory')
    plt.ylabel('Frequency')
    print('\tMin: {min}\n\tMax: {max}\n\tMean: {mean:.1f}\n\tMedian: {median}\n'.format(
        min=min(diversities),
        max=max(diversities),
        mean=np.mean(diversities),
        median=int(np.median(diversities))))


# how to select milestones?
# idea: just take the most frequently visited microstate from each trajectory
def identify_cores_by_frequency(dtrajs):
    '''
    list of dtrajs --> non-redundant list of indices of the most populous trial milestones in each dtraj
    '''
    return list(set([int(np.median(dtraj)) for dtraj in dtrajs]))


# what's the smallest set of cores such that each core appears at least once in each trajectory?
def compute_occurrence_matrix(dtrajs):
    '''
    computes an n_trajs x n_states binary matrix, indicating which trajectories contain which states

    occurrence_matrix[i,j] = 1 if j in dtrajs[i]
                             0 otherwise
    '''
    all_elements = set(np.hstack(dtrajs))
    n_trajs = len(dtrajs)
    n_states = len(all_elements)

    # n_trajs x n_states binary matrix, indicating which trajectories contain which states
    occurrence_matrix = np.zeros((len(dtrajs), n_states))
    for i in range(len(dtrajs)):
        occurrence_matrix[i] = np.in1d(np.arange(n_states), dtrajs[i])
    return occurrence_matrix

def metastability_index_using_comprehensions(gamma, M):
    '''
    possibly buggy, since this is giving infs a lot of the time

    numerator = max_{i \in M} (max_{j \in M \minus i} Gamma[i,j])
    denominator = min_{i \not\in M} (max_{j \in M} Gamma[i,j])
    '''
    not_M = set(range(len(gamma))) - set(M)
    M_minus = lambda i: set(M) - set([i])

    numerator = max([max([gamma[i, j] for j in M_minus(i) if (i != j)]) for i in M])
    denominator = min([max([gamma[i, j] for j in M if (i != j)]) for i in not_M])
    return numerator / denominator

def metastability_index(gamma, M):
    numerator = 0
    for i in M:
        # maximize over i
        Mprime = set(M) - set([i])
        for j in Mprime:
            # maximize over j
            if gamma[i, j] > numerator:
                numerator = gamma[i, j]

    denominators = []
    for i in set(range(len(gamma))) - set(M):

        # maximize over j
        denominator = 0
        for j in M:
            if gamma[i, j] > denominator:
                denominator = gamma[i, j]

        denominators.append(denominator)

    # minimize over i
    denominator = min(denominators)

    return numerator / denominator

# attempt to implement the optimizer from "Optimized Markov State Models
# for Metastable Systems"
def get_j_dagger(gamma, i):
    '''
    j_dagger(i) = argmax_{k=/=i} Gamma_ik

    eq. 8 : the index of the trial milestone that the trajectory is most likely to hit
    '''

    return np.argmax(gamma[i])

def update_M(gamma, M):
    '''
    M --> {j_dagger(i) | i \in M}
    '''
    return set([get_j_dagger(gamma, i) for i in M])

def iterate_M(gamma, init_M, max_iter=20):
    '''
    perform the `update_M` iteration max_iter times
    '''
    Ms = [init_M]
    for i in range(max_iter):
        Ms.append(update_M(gamma, Ms[-1]))
    return Ms

def assign_microstates_to_macrostates(gamma, milestones):
    '''
    assign microstate i to the milestone it is most likely to hit, according to gamma

    (renumbers so that c)
    '''

    # convert milestones to index array, if it isn't already one
    if type(milestones) != np.ndarray: milestones = np.array(list(milestones))

    # column-slice
    gamma_to_milestones = gamma[:, milestones]

    # get assignment[i] = argmax_j gamma_to_milestones[i, j]
    assignments = np.argmax(gamma_to_milestones, 1)

    # make sure milestone microstates are assigned properly
    for i in range(len(milestones)):
        gamma_to_milestones[milestones[i]] = i

    return assignments

def apply_coarse_graining(dtrajs, assignment):
    assign = np.vectorize(lambda i: assignment[i])
    return [assign(dtraj) for dtraj in dtrajs]

def estimate_msm(dtrajs, lag=50):
    return pyemma.msm.estimate_markov_model(dtrajs, lag=lag)

def random_milestone_set_baseline(dtrajs_src, dtrajs_abl):
    '''
    what's the metastability of a coarse-graining derived from a random milestone set?

    plots histograms, returns coarse-grained MSMs

    Parameters
    ----------
    dtrajs_src
    dtrajs_abl

    Returns
    -------
    ms_msms_src
    ms_msms_abl
    '''
    ms_msms_src = []
    ms_msms_abl = []

    n_samples = 100
    n_macrostates = 50

    inds = np.arange(1000)

    for _ in tqdm(range(n_samples)):
        np.random.shuffle(inds)

        ms_dtrajs = apply_coarse_graining(dtrajs_src,
                                          assignment=assign_microstates_to_macrostates(gamma_src, inds[:n_macrostates]))
        ms_msms_src.append(estimate_msm(ms_dtrajs))

        ms_dtrajs = apply_coarse_graining(dtrajs_abl,
                                          assignment=assign_microstates_to_macrostates(gamma_abl, inds[:n_macrostates]))
        ms_msms_abl.append(estimate_msm(ms_dtrajs))


    n_bins = 10
    plt.hist([np.trace(m.P) for m in ms_msms_src], bins=n_bins, alpha=0.5, histtype='stepfilled', label='Src')
    plt.hist([np.trace(m.P) for m in ms_msms_abl], bins=n_bins, alpha=0.5, histtype='stepfilled', label='Abl')
    plt.xlim(0, n_macrostates)
    plt.legend(loc='best')
    plt.title('Metastability of 50-state coarse-grainings\nusing $\Gamma$ and random milestone sets')
    plt.xlabel('Metastability')
    plt.ylabel('Occurrence')

    plt.savefig('random_milestone_baseline.png', dpi=300)
    plt.close()

    return ms_msms_src, ms_msms_abl

# as a baseline, what's the metastability if we just do random coarse-graining?
def random_cg_baseline(dtrajs_src, dtrajs_abl):
    '''
    what's the metastability of a coarse-graining derived from a random milestone set?

    plots histograms, returns coarse-grained MSMs

    Parameters
    ----------
    dtrajs_src
    dtrajs_abl

    Returns
    -------
    cg_msms_src
    cg_msms_abl

    '''
    cg_msms_src = []
    cg_msms_abl = []

    n_samples = 100
    n_macrostates = 50

    for _ in tqdm(range(n_samples)):
        cg_dtrajs = apply_coarse_graining(dtrajs_src, assignment=np.random.randint(0, n_macrostates, 1000))
        cg_msms_src.append(estimate_msm(cg_dtrajs))

        cg_dtrajs = apply_coarse_graining(dtrajs_abl, assignment=np.random.randint(0, n_macrostates, 1000))
        cg_msms_abl.append(estimate_msm(cg_dtrajs))

    n_bins = 10
    plt.hist([np.trace(m.P) for m in cg_msms_src], bins=n_bins, alpha=0.5, histtype='stepfilled', label='Src')
    plt.hist([np.trace(m.P) for m in cg_msms_abl], bins=n_bins, alpha=0.5, histtype='stepfilled', label='Abl')
    plt.xlim(0, n_macrostates)
    plt.legend(loc='best')
    plt.title('Metastability of 50-state coarse-grainings\n using uniform random assignments')
    plt.xlabel('Metastability')
    plt.ylabel('Occurrence')

    plt.savefig('uniform_random_cg_baseline.png', dpi=300)
    plt.close()

    return cg_msms_src, cg_msms_abl

# what about just directly minimizing the metastability index?

# the metastability index is composed of a numerator and a denominator term
# we want the numerator to be small and the denominator to be large

# take a step to try to decrease the numerator
def remove_worst_using_numerator(gamma, M):
    '''
    identifies i the worst trial microstate currently in M, and removes it

    M <-- M \ i

    where i is the element of M for which the probability
    is the highest that, after hitting i, the trajectory
    will hit some other target milestone j \in M before hitting i
    again
    '''
    M = list(M)
    i = np.argmax([max(gamma[i, np.array(list(set(M) - set([i])))]) for i in M])
    _ = M.pop(i)
    return set(M)

# take a step to try to increase the denominator
def add_best_using_denominator_term(gamma, M):
    '''
    identifies i the best trial microstate not currently in M, the denominator of metastability_index

    M <-- M + {i}
    '''
    l = list(set(range(len(gamma))) - set(M))
    i = np.argmin([max(gamma[i]) for i in l])

    return set(M).union(set([i]))

def greedy_subtraction_optimizer(gamma, target_size=500, callback=None):
    '''
    at each iteration, remove the "worst" milestone identified in the numerator of the metastability index
    '''
    M = set(range(len(gamma)))
    for _ in tqdm(range(len(M) - target_size)):
        M = remove_worst_using_numerator(gamma, M)
        if callback: callback(M)
    return M

def milestone_dtraj(dtraj, milestoning_set):
    milestoned_traj = np.zeros(len(dtraj), dtype=int)

    last_milestone_visited = -1

    dtraj_in_milestoning_set = np.in1d(dtraj, np.array(list(milestoning_set)))

    for i in range(len(dtraj)):
        if dtraj_in_milestoning_set[i]:
            # if dtraj[i] in milestoning_set:
            last_milestone_visited = dtraj[i]
        milestoned_traj[i] = last_milestone_visited
    return milestoned_traj

def milestone_serial(dtrajs, milestoning_set, n_threads=10):
    '''
    construct milestone-based trajectories
    (serial -- reference)
    '''
    m_func = partial(milestone_dtraj, milestoning_set=milestoning_set)
    return map(m_func, dtrajs)

def milestone(dtrajs, milestoning_set, n_threads=10):
    '''
    construct milestone-based trajectories
    (parallel -- use this version)
    '''
    p = Pool(n_threads)
    m_func = partial(milestone_dtraj, milestoning_set=milestoning_set)
    result = p.map(m_func, dtrajs)
    del (p)
    return result

# what if we try PageRank?
def gamma_to_graph(gamma):
    '''
    convert the gamma matrix to a directed graph where gamma[i,j] is the weight on the edge i->j
    '''
    graph = nx.DiGraph()
    for i in range(len(gamma)):
        for j in range(len(gamma)):
            if gamma[i, j] != 0:
                graph.add_edge(i, j, weight=gamma[i, j])
    return graph

def get_pagerank(graph):
    '''
    given a graph, return an array where array[i] = pagerank(node i)

    Note
    ----
    power iteration often failed on src / abl when using plain old nx.pagerank,
    so using nx.pagerank_numpy (which uses NumPy's interface to the LAPACK eigenvalue solvers).
    '''
    pagerank_dict = nx.pagerank_numpy(graph)
    pagerank = np.array([pagerank_dict[i] for i in range(len(pagerank_dict))])
    return pagerank

# plot coarse-grained free energies
def plot_free_energies(msm, name=None):
    ''' given an msm, plot the estimated free energies of '''
    f_i = -np.log(sorted(msm.stationary_distribution))[::-1]
    f_i -= f_i.min()
    plt.plot(f_i, '.', label=name)
    plt.hlines(6, 0, len(f_i))

# greedy addition optimizer
def greedy_addition_optimizer(gamma, init_M, target_size=150, callback=None):
    '''
    at each iteration, add the "best" trial milestone identified in the denominator of the metastability index
    '''
    M = init_M
    for _ in tqdm(range(target_size - len(init_M))):
        M = add_best_using_denominator_term(gamma, M)
        if callback: callback(M)
    return M

# how long are the strings of zeros?
def get_between_core_waiting_times(binarized):
    '''
    given a list of binary 1d arrays, get all the strings of consecutive zeros, and return a list of their lengths
    '''

    waits = []
    current_wait = 0
    for b in binarized:
        for i in b:
            if not i:
                current_wait += 1
            elif i and current_wait > 0:
                waits.append(current_wait)
                current_wait = 0
    return waits

def count_transitions(milestoned_dtrajs):
    '''
    accumulate the matrix N_{i,j}
    '''
    n = np.max(np.hstack(milestoned_dtrajs)) + 1
    N = np.zeros((n, n))
    for dtraj in milestoned_dtrajs:
        for i in range(len(dtraj) - 1):
            if dtraj[i] != dtraj[i + 1]:
                N[dtraj[i], dtraj[i + 1]] += 1
    return N

def count_dwell_times(milestoned_dtrajs):
    return np.bincount(np.hstack(milestoned_dtrajs))


def trim_milestoned_dtrajs(milestoned_dtrajs):
    '''
    remove the beginning segments, before any core has been hit

    remove the ending segments, for which the waiting times are undetermined
    '''

    trimmed_milestoned_dtrajs = []
    for dtraj in milestoned_dtrajs:

        # start at the first index that isn't -1
        start_ind = np.argmax(dtraj != -1)

        # this will miss the case where the whole dtraj is -1's, so we'll catch that here:
        if np.sum(dtraj != -1) == 0: start_ind = len(dtraj)

        # end before the last waiting segment, since its waiting time is undetermined
        end_state = dtraj[-1]
        end_ind = len(dtraj) - (np.argmax(dtraj[::-1] != end_state))

        # this will miss the case where the whole dtraj is the same non-minus-1 state, so we'll catch that here:
        if len(set(dtraj)) < 2: end_ind = 0

        # if this doesn't trim away the whole trajectory, trim and add it to the list
        if end_ind > start_ind: trimmed_milestoned_dtrajs.append(dtraj[start_ind:end_ind])

    return trimmed_milestoned_dtrajs

def maximum_likelihood_milestoning_estimator(processed_dtrajs):
    '''
    assume the input dtrajs have been processed already, i.e. trimmed and mapped
    \hat{k}_{i,j} \equiv \frac{N_{i,j}}{R_i}
    '''
    # compute transition counts
    N = count_transitions(processed_dtrajs)
    # compute waiting times
    R = count_dwell_times(processed_dtrajs)

    # return N_ij / R_i
    return (N.T / R).T

def map_dtrajs(dtrajs):
    '''
    given dtrajs with N possibly non-consecutive indices, map them to range(N) indices
    '''

    # flatten list of dtrajs
    dtrajs_ = np.hstack(dtrajs)

    # assuming things start at 0, how many skipped indices are there?
    n_slots = max(dtrajs_) + 1
    observed_states = sorted(list(set(dtrajs_)))
    n_states = len(observed_states)

    # give each observed state an index between 0 and n_states
    state_map = np.zeros(n_slots)
    for i in range(len(observed_states)):
        state_map[observed_states[i]] = i

    # apply the map
    nice_dtrajs = [np.array([state_map[dtraj[i]] for i in range(len(dtraj))], dtype=int) for dtraj in dtrajs]

    return nice_dtrajs, state_map


def sample_milestoning_rate_matrices(milestoned_dtrajs, n_samples=100):
    '''
    k_ij \sim Gamma(shape=1/R_i, scale=N_ij + 1)
    '''
    length = len(np.hstack(milestoned_dtrajs))

    R = 1.0 * count_dwell_times(milestoned_dtrajs)
    N = 1.0 * count_transitions(milestoned_dtrajs)

    n = len(R)
    shape = 1.0 / np.vstack([R] * n).T
    scale = N + 1
    return np.random.gamma(shape=shape, scale=scale, size=(n_samples, n, n))

# implement MCSA with the 3 move types
def MCSA(init_M, gamma, schedule, n_restarts=100, callback=None):
    '''
    stochastically optimize the metastability index, given an initial estimate
    '''
    raise(NotImplementedError())


if __name__ == '__main__':
    # get dtrajs from feature-based clustering
    dtrajs_src = np.load('contacts-results/src_10471_dtrajs.npy')
    dtrajs_abl = np.load('contacts-results/abl_10472_dtrajs.npy')

    # get dtrajs from RMSD
    dtrajs_src_rmsd = [np.load(f) for f in glob('rmsd_clustering_results/10471/dtrajs/*.npy')]
    dtrajs_abl_rmsd = [np.load(f) for f in glob('rmsd_clustering_results/10472/dtrajs/*.npy')]

    # compute gamma matrices
    gamma_src = estimate_gamma_version_1(dtrajs_src)
    gamma_abl = estimate_gamma_version_1(dtrajs_abl)

    # visually inspect gamma_abl and gamma_src

    plt.figure()
    plt.imshow(gamma_src, interpolation='none', cmap='Blues')
    plt.title('Src')
    plt.savefig('gamma_src.png', dpi=300)
    plt.close()

    plt.figure()
    plt.imshow(gamma_abl, interpolation='none', cmap='Blues')
    plt.title('Abl')
    plt.savefig('gamma_abl.png', dpi=300)
    plt.close()

    # the gamma matrices look pretty sparse
    for (name, matrix) in [('src', gamma_src), ('abl', gamma_abl)]:
        print('{0} : sparsity = {1:.3f}%'.format(name, 100.0 * np.sum(matrix == 0) / np.prod(matrix.shape)))

    # plot a single row of the gamma matrix
    plt.figure()
    plt.plot(gamma_src[0])
    plt.title('$\Gamma[0,:]$ src')
    plt.savefig('gamma_0_src.png',dpi=300)
    plt.close()

    # how many different microstates are visited in each trajectory?
    for (name, dtrajs) in [('src_contacts', dtrajs_src),
                           ('abl_contacts', dtrajs_abl),
                           ('src_rmsd', dtrajs_src_rmsd),
                           ('abl_rmsd', dtrajs_abl_rmsd)]:
        print(name)
        plt.figure()
        get_diversities(dtrajs)
        plt.title(name)
        plt.savefig('{0}_diversities.png'.format(name), dpi=300)
        plt.close()

    # try a super simple heuristic for identifying cores
    print('Trying "most-populous" heuristic for identifying cores')
    cores_src_rmsd = identify_cores_by_frequency(dtrajs_src_rmsd)
    print('src_rmsd: {0} cores from {1} trajectories'.format(len(cores_src_rmsd), len(dtrajs_src)))

    cores_abl_rmsd = identify_cores_by_frequency(dtrajs_abl_rmsd)
    print('abl_rmsd: {0} cores from {1} trajectories'.format(len(cores_abl_rmsd), len(dtrajs_abl)))

    cores_src = identify_cores_by_frequency(dtrajs_src)
    print('src_contacts: {0} cores from {1} trajectories'.format(len(cores_src), len(dtrajs_src)))

    cores_abl = identify_cores_by_frequency(dtrajs_abl)
    print('abl_contacts: {0} cores from {1} trajectories'.format(len(cores_abl), len(dtrajs_abl)))

    # examine the microstate vs. trajectory occurrence matrix
    occurrence_matrix = compute_occurrence_matrix(dtrajs_src)

    # plot it
    plt.figure()
    plt.imshow(occurrence_matrix.T, interpolation='none', cmap='Blues')
    plt.ylabel('State index')
    plt.xlabel('Trajectory')
    plt.title('Which states appear in which trajectories?\n(Src contacts-based clustering)')
    plt.savefig('src_contacts_occurrence_matrix.png')
    plt.close()

    # for rmsd clustering
    occurrence_matrix = compute_occurrence_matrix(dtrajs_src_rmsd)

    # how many states appear in only 1 trajectory?
    print(sum(occurrence_matrix.sum(0) == 1))

    # how many trajectories contain only 1 state?
    print(sum(occurrence_matrix.sum(1) == 1))

    # for contacts-based clustering
    occurrence_matrix = compute_occurrence_matrix(dtrajs_src)
    # how many states appear in only 1 trajectory?
    print(sum(occurrence_matrix.sum(0) == 1))

    # how many trajectories contain only 1 state?
    print(sum(occurrence_matrix.sum(1) == 1))

    # which states only appear in only 1 trajectory?
    indices = np.arange(len(occurrence_matrix.T))[occurrence_matrix.sum(0) == 1]

    # how many trajectories contain any of these?
    sum([np.in1d(dtraj, indices).sum() > 0 for dtraj in dtrajs_src])

    # to-do: move analysis of occurrence matrix into a separate function?


    # apply optimizer from paper

    # example optimization trace:
    Ms = iterate_M(gamma_src, init_M=set(range(len(gamma_src))))
    plt.plot([len(M) for M in Ms])
    plt.figure()
    plt.plot([metastability_index(gamma_src, M) for M in Ms[1:]])
    plt.title('Metastability index during optimization\n(Src, initialized from full trial milestone set)')
    plt.savefig('src_metastability_index_trace.png',dpi=300)
    plt.close()

    # what does this iteration look like?
    plt.figure()
    inds = np.arange(len(gamma_src))
    all_points = []
    points = inds[:20]
    plt.scatter(np.zeros(len(points)), points)
    n_iter = 10
    for i in range(n_iter):
        all_points.append(points)
        points = [get_j_dagger(gamma_src, t) for t in points]
        plt.scatter(np.ones(len(points)) * (i + 1), points)
    all_points = np.array(all_points)
    plt.plot(all_points);

    plt.title('Optimization')

    # initialized from inds 0-20
    Ms = iterate_M(gamma_src, init_M=set(inds[:20]))
    plt.plot([len(M) for M in Ms])
    plt.figure()
    plt.plot([metastability_index(gamma_src, M) for M in Ms])

    # what do these optimization traces look like?
    n_milestones = 20
    n_samples = 100

    np.random.seed(0)
    inds = np.arange(len(gamma_src))

    traces = []

    for _ in tqdm(range(n_samples)):
        np.random.shuffle(inds)
        Ms = iterate_M(gamma_src, init_M=set(inds[:n_milestones]))
        plt.plot([metastability_index(gamma_src, M) for M in Ms])
        traces.append(Ms)
    plt.xlabel('Iteration')
    plt.ylabel('Metastability index (lower is better)')
    plt.title('Metastability-index optimization trace:\nInitialized 100 times with size-20 milestone sets')

    # In[17]:

    # what values of this index do we get if we pick milestones uniformly at random?
    # how much better do we do on average after applying the update?

    # before
    plt.hist([metastability_index(gamma_src, Ms[0]) for Ms in traces], bins=50, alpha=0.5, histtype='stepfilled',
             label='initial');

    # after
    plt.hist([metastability_index(gamma_src, Ms[-1]) for Ms in traces], bins=50, alpha=0.5, histtype='stepfilled',
             label='final');

    # plot labels
    plt.legend(loc='best')
    plt.ylabel('occurrence')
    plt.xlabel('metastability index (lower is better)')
    plt.title('effect of j_dagger update')

    # hmm, something must be going wrong: this update doesn't actually appear
    # to improve the metastability index!

    # how about another graph-based heuristic to select "cores?"
    graph_src = gamma_to_graph(gamma_src)
    graph_abl = gamma_to_graph(gamma_abl)

    page_rank_src = get_pagerank(graph_src)
    page_rank_abl = get_pagerank(graph_abl)

    plt.hist(page_rank_src, bins=50, alpha=0.5);
    plt.hist(page_rank_abl, bins=50, alpha=0.5);
    plt.xlabel('PageRank')
    plt.ylabel('Frequency')
    plt.title('Distribution of PageRank scores in in Src and Abl $\Gamma$ graphs')
    plt.savefig('')

    # what if we just take milestones greedily according to their PageRank scores?
    pagerank_milestones_src = np.argsort(page_rank_src)[::-1][:n_milestones]
    metastability_index(gamma_src, pagerank_milestones_src)

    # let's try the direct, greedy subtraction optimization approach
    n_macrostates = 50

    # define callbacks to store intermediate results
    Ms_src = []
    metastability_indices_src = []

    Ms_abl = []
    metastability_indices_abl = []

    def callback_src(M):
        try:
            Ms_src.append(M)
            metastability_indices_src.append(metastability_index(gamma_src, M))
        except:
            print('Something went wrong!')

    def callback_abl(M):
        try:
            Ms_abl.append(M)
            metastability_indices_abl.append(metastability_index(gamma_abl, M))
        except:
            print('Something went wrong!')

    M_src = greedy_subtraction_optimizer(gamma_src, callback=callback_src)
    M_abl = greedy_subtraction_optimizer(gamma_abl, callback=callback_abl)

    plt.plot(metastability_indices_src, label='Src')
    plt.plot(metastability_indices_abl, label='Abl')

    plt.title('Optimization traces:\ngreedy milestone removal')
    plt.ylabel('Metastability index (lower is better)')
    plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig('greedy_milestone_subtraction_traces.png', dpi=300)
    plt.close()

    # let's try the direct greedy addition approach
    np.random.seed(0)
    init_M = set(np.random.randint(0, 1000, 20))
    Ms_src = []
    metastability_indices_src = []

    Ms_abl = []
    metastability_indices_abl = []

    M_src = greedy_addition_optimizer(gamma_src, init_M, callback=callback_src)
    M_abl = greedy_addition_optimizer(gamma_abl, init_M, callback=callback_abl)

    # plot metastability index vs. iteration
    plt.plot(metastability_indices_src, label='Src')
    plt.plot(metastability_indices_abl, label='Abl')
    plt.title('Optimization traces:\ngreedy milestone addition')
    plt.ylabel('Metastability index (lower is better)')
    plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig('greedy_milestone_addition_traces.png', dpi=300)
    plt.close()

    # let's look at the resulting coarse-grained MSMs
    cg_dtrajs = apply_coarse_graining(dtrajs_src, assignment=assign_microstates_to_macrostates(gamma_src, M_src))
    cg_msm_src = estimate_msm(cg_dtrajs)
    print(np.trace(cg_msm_src.P))

    # and again, for abl
    cg_dtrajs = apply_coarse_graining(dtrajs_abl, assignment=assign_microstates_to_macrostates(gamma_abl, M_abl))
    cg_msm_abl = estimate_msm(cg_dtrajs)
    print(np.trace(cg_msm_abl.P))

    # plot free energies of resulting macrostates
    plt.figure()
    plot_free_energies(cg_msm_src, name='Src')
    plot_free_energies(cg_msm_abl, name='Abl')
    plt.legend(loc='best')
    plt.xlabel('Macrostate')
    plt.ylabel(r'$\Delta G$ $(k_B T)$')
    plt.title('Macrostate free energies after greedy milestone addition')
    plt.savefig('greedy_milestone_addition_free_energies.png', dpi=300)
    plt.close()

    # core hitting times: how long do we have to wait after we leave a core to hit another core?
    dtrajs = dtrajs_src
    cores =  M_src
    binarized = [np.in1d(dtraj, cores) for dtraj in dtrajs]
    # one of the assumptions of this approach is that, if we start outside a core, we'll hit _some_ core very quickly.
    # what do the actual between-core waiting times look like?
    waits = get_between_core_waiting_times(binarized)
    plt.hist(waits, bins=50);
    plt.yscale('log')
    plt.title('Between-core waiting times')
    plt.savefig('between_core_waiting_times.png', dpi=300)
    plt.close()

    # should we also do this individually for every milestone in the milestone set?

    # how many of the trajectories don't actually ever hit any of the milestones?
    sum([np.sum(b) == 0 for b in binarized]), len(binarized)

    # how much is left after trimming?
    milestoned_dtrajs = milestone(dtrajs_src, cores_src)
    trimmed_milestoned_dtrajs = trim_milestoned_dtrajs(milestoned_dtrajs)
    print('src_contacts, fraction of frames retained after trimming', 1.0 * sum([len(d) for d in trimmed_milestoned_dtrajs]) / sum([len(d) for d in milestoned_dtrajs]))

    # map the discrete state indices to consecutive integers in range(0,n_states)
    nice_dtrajs, state_map = map_dtrajs(trimmed_milestoned_dtrajs)

    # estimate milestoning rates
    MLE = maximum_likelihood_milestoning_estimator(nice_dtrajs)
    samples = sample_milestoning_rate_matrices(nice_dtrajs, n_samples=1000)

    # plot mean of sampled rate matrices
    plt.imshow(np.mean(samples, 0), interpolation='none', cmap='Blues');
    plt.colorbar()
    plt.savefig('rate_matrix_sample_mean.png', dpi=300)
    plt.close()

    # plot MLE of rate matrix
    plt.figure()
    plt.imshow(MLE, interpolation='none', cmap='Blues');
    plt.colorbar()
    plt.title('MLE')
    plt.savefig('rate_matrix_mle.png', dpi=300)
    plt.close()


    cg_dtrajs = apply_coarse_graining(dtrajs_src, assignment=assign_microstates_to_macrostates(gamma_src, Ms_src[-1]))
    cg_msm_src = estimate_msm(cg_dtrajs)
    np.trace(cg_msm_src.P)

    # and again, for abl
    cg_dtrajs = apply_coarse_graining(dtrajs_abl, assignment=assign_microstates_to_macrostates(gamma_abl, Ms_abl[-1]))
    cg_msm_abl = estimate_msm(cg_dtrajs)
    np.trace(cg_msm_abl.P)

    # plot and compare their free energies
    plot_free_energies(cg_msm_src, name='Src')
    plot_free_energies(cg_msm_abl, name='Abl')
    plt.legend(loc='best')
    plt.xlabel('Macrostate')
    plt.ylabel(r'$\Delta G$ $(k_B T)$')
    plt.title('Macrostate free energies')