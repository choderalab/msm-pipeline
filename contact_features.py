import mdtraj as md
import numpy as np

def find_respairs_that_changed(fnames,
                               scheme = 'ca',    # or 'closest' or 'closest-heavy'
                               threshold = 0.4,
                               stride = 100,
                               max_respairs = 1000):
    '''

    Parameters
    ----------
    fnames : list of paths to trajectories

    scheme : 'ca' or 'closest' or 'closest-heavy'

    threshold : float
        contact threshold (nm)
    '''
    distances = []
    for fname in fnames:
        traj = md.load(fname,stride=stride)
        pairwise_distances,residue_pairs = md.compute_contacts(traj,scheme=scheme)
        distances.append(pairwise_distances)
    distances = np.vstack(distances)

    # identify contacts that change by counting how many times the distances were
    # greater than and less than the threshold
    num_times_greater_than = (distances>threshold).sum(0)
    num_times_less_than = (distances<threshold).sum(0)
    changed = (num_times_greater_than > 0) * (num_times_less_than > 0)
    print("Number of contacts that changed: {0}".format(changed.sum()))
    print("Total number of possible contacts: {0}".format(len(residue_pairs)))

    if len(changed) > max_respairs:
        n_diff = np.min(np.vstack((num_times_less_than,num_times_greater_than)),0)
        indices = sorted(np.arange(len(n_diff)),key=lambda i:-n_diff[i])
        changed = indices[:max_respairs]

    # now turn this bitmask into a list of relevant residue pairs
    respairs_that_changed = residue_pairs[changed]

    return respairs_that_changed
