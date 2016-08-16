import numpy as np
import numpy.random as npr

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
    X = [npr.randn(1000,100) for _ in range(10)]
    import pyemma
    tica = pyemma.coordinates.tica(X)
    tica_output = tica.get_output()

    kmeans = pyemma.coordinates.cluster_kmeans(tica_output, k = 100)
    dtrajs = [dtraj.flatten() for dtraj in kmeans.get_output()]
    msm = pyemma.msm.estimate_markov_model(dtrajs, 1)

    return dtrajs, tica, tica_output, msm

def main():
    dtrajs, tica, tica_output, msm = generate_test_instance()

    from generate_report import make_plots, compute_its
    compute_its(dtrajs, project_name = 'test')
    make_plots(dtrajs, tica, tica_output, msm, project_name = 'test')

if __name__ == '__main__':
    main()
