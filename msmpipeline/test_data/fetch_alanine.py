from msmbuilder.example_datasets import AlanineDipeptide
trajs = AlanineDipeptide().get().trajectories
for i,traj in enumerate(trajs):
    traj.save_hdf5('alanine_{0}.h5'.format(i))