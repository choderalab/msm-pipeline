from multiprocessing import Pool
from functools import partial
import mdtraj as md
import numpy as np

def get_distances(fname, scheme, stride):
    '''
    Function callable by a multiprocessing Pool
    
    Parameters
    ----------
    fname : string
        filename of trajectory
    scheme : string
        'ca' or 'closest' or 'closest-heavy'
    stride : int
        thinning factor: only look at every `stride`th frame
        
    Returns
    -------
    pairwise_distances : numpy array
    
    residue_pairs : list of tuples
    '''
    traj = md.load(fname, stride = stride)
    pairwise_distances,residue_pairs = md.compute_contacts(traj, scheme = scheme)
    return pairwise_distances, residue_pairs

def find_respairs_that_changed(fnames,
                               scheme = 'ca',    # or 'closest' or 'closest-heavy'
                               threshold = 0.4,
                               stride = 100,
                               max_respairs = 1000,
                               num_threads = 12):
    '''
    Finds all the residue pairs that crossed `threshold` at least once in a strided subset of 
    trajectory data. If that number of features is still too high, just take the top `max_respairs`
    most frequently threshold-crossing residue pairs.
    
    Parameters
    ----------
    fnames : list of strings
        paths to trajectories

    scheme : string
        how to define the distance between two residues.
        'ca' or 'closest' or 'closest-heavy' (alpha-carbon, closest atom, or closest heavy atom)

    threshold : float
        contact threshold (nm)
    
    stride : int
        thinning factor: only look at every `stride`th frame
        
    max_respairs : int
        maximum number of features to return
        
    num_threads : int
        number of threads to use 
    
    Returns
    -------
    respairs_that_changed : list of tuples
        each element of the list is a length-2 tuple
    '''
    # parallelize over fnames
    pool = Pool(num_threads)
    get_distances_ = partial(get_distances, scheme=scheme, stride=stride)
    distances = pool.map(get_distances_, fnames)
    residue_pairs = distances[0][1]
    distances = np.vstack([d[0] for d in distances])

    # identify contacts that change by counting how many times the distances were
    # greater than and less than the threshold
    num_times_greater_than = (distances > threshold).sum(0)
    num_times_less_than = (distances < threshold).sum(0)
    changed = (num_times_greater_than > 0) * (num_times_less_than > 0)
    print("Number of contacts that changed: {0}".format(changed.sum()))
    print("Total number of possible contacts: {0}".format(len(residue_pairs)))

    if len(changed) > max_respairs:
        n_diff = np.min(np.vstack((num_times_less_than, num_times_greater_than)), 0)
        indices = sorted(np.arange(len(n_diff)), key = lambda i: -n_diff[i])
        changed = indices[:max_respairs]

    # now turn this bitmask into a list of relevant residue pairs
    respairs_that_changed = residue_pairs[changed]

    return respairs_that_changed
