import mdtraj as md

import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyemma

from generate_report import make_plots

msm_lag=50
max_tics = 500
dtrajs = np.load('mtor/mTORKinase_dtrajs.npy')
dtrajs = [dtraj for dtraj in dtrajs]
Y = np.load('mtor/mTORKinase_tica_projection.npy')
project_name = 'mTORKinase'
msm = pyemma.msm.estimate_markov_model(dtrajs, lag=msm_lag)


make_plots(dtrajs, tica_output=Y, msm=msm, project_name=project_name)