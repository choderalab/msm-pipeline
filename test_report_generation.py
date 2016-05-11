def generate_test_instance():
    from msmbuilder.example_datasets import MetEnkephalin
    trajs = MetEnkephalin().get().trajectories
    from msmbuilder.featurizer import DihedralFeaturizer
    X = DihedralFeaturizer().fit_transform(trajs)
    import pyemma
    tica = pyemma.coordinates.tica(X)
    tica_output = tica.get_output()

    kmeans = pyemma.coordinates.cluster_kmeans(tica_output,k=100)
    dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]
    msm = pyemma.msm.estimate_markov_model(dtrajs,1)

    return dtrajs, tica, tica_output,msm

dtrajs,tica,tica_output = generate_test_instance()

from generate_report import make_plots,compute_its
compute_its(dtrajs,project_name='test')
make_plots(dtrajs,tica,tica_output,msm,project_name='test')
