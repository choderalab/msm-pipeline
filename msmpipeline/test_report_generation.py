def generate_test_instance():
    '''
    Fetches example data from `msmbuilder.example_datasets`, then clusters w.r.t.
    a tICA-derived kinetic distance, and returns results.
    
    Returns
    -------
    dtrajs : list of int-arrays
    
    tica : pyemma tICA object
    
    tica_output : list of numpy.ndarrays
    
    msm : pyemma MSM object
    '''
    from msmbuilder.example_datasets import MetEnkephalin
    trajs = MetEnkephalin().get().trajectories
    from msmbuilder.featurizer import DihedralFeaturizer
    X = DihedralFeaturizer().fit_transform(trajs)
    import pyemma
    tica = pyemma.coordinates.tica(X)
    tica_output = tica.get_output()

    kmeans = pyemma.coordinates.cluster_kmeans(tica_output, k = 100)
    dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]
    msm = pyemma.msm.estimate_markov_model(dtrajs, 1)

    return dtrajs, tica, tica_output, msm

if __name__ == '__main__':
    dtrajs,tica,tica_output = generate_test_instance()
    
    from generate_report import make_plots, compute_its
    compute_its(dtrajs, project_name = 'test')
    make_plots(dtrajs, tica, tica_output, msm, project_name = 'test')
