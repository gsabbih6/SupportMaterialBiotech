# This script was used for thesis work and other publications to prepare molecular system for REMD simulations and to calculate the DFE from the Gaussians.

# @Author Godfred Oppong Sabbih gsabbih5@gmail.com or fzp281@mocs.utc.edu
# By using this file you agree to sight the relevant publications
# Computational generation and characterization of IsdA-binding aptamers with single-molecule FRET analysis
# Godfred O. Sabbih1, Kalani M. Wijesinghe2, Chamika Algama2, Soma Dhakal2 and Michael K. Dan-quah1*


# don't forget to use openmpi/gcc/3.1
import os, click
import pickle
import subprocess
from math import log
import MDAnalysis as mda
import seaborn as sn

import matplotlib
import numpy as np

import pandas as pd
from symfit import parameters, variables, sin, cos, Fit, Piecewise, exp, Eq, Model

matplotlib.use('Agg')
var = ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo',
       'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg

boltzman_constant = 0.001985875  # kcal/mol.K
temp = 310.15  # K


def prepare_plumed_config(reference_pdb, pro, nuc):
    with open('plumed.dat', 'w') as f:
        f.write(
            'MOLINFO STRUCTURE={} PYTHON_BIN=python\n'
            '# this is needed to allow arbitrary pairs to try exchanges\n'
            '# in this case, 0<->1, 0<->2, and 1<->2\n'
            'RANDOM_EXCHANGES\n'
            'nuc_back: CENTER ATOMS={}\n'
            'pro_back: CENTER ATOMS={}\n'
            'dist: DISTANCE ATOMS=nuc_back,pro_back\n'
            '# You can use the same parameters that you used in masterclass 21.4\n'
            'm: METAD ARG=dist SIGMA={} HEIGHT={} PACE={} GRID_MIN=-15 GRID_MAX=15\n'
            'uwall: UPPER_WALLS ARG=dist AT=6.0 KAPPA=150.0 EXP=2 EPS=1 OFFSET=0\n'
            'PRINT STRIDE=5 ARG=dist,m.bias FILE=../colvar_replica.dat'
            .format(reference_pdb, nuc, pro, 0.05, 0.01, 500)
        )


def prep_replicas(window_size):
    for i in range(window_size):
        if os.path.exists(os.getcwd() + '/' + 'replica' + str(i)):
            subprocess.run("rm -r replica{}".format(i), shell=True)
        subprocess.run("mkdir replica{} && \
                                   cp step5_1.tpr replica{} && cp step3_input.pdb replica{}".format(i, i, i),
                       shell=True)

# use plumed to estimate fes
def calculateFES(bias_hills_path):
    
    os.chdir(bias_hills_path)
    count = bias_hills_path.split('replica')[-1]
    print(count)
    print(bias_hills_path)
    print(os.getcwd())
    os.system('plumed sum_hills --hills HILLS.{}'.format(count))
    fes_df = pd.read_table(bias_hills_path + '/fes.dat', skiprows=5, delimiter=r"\s+",
                           names=['Dist(nm)', 'FES(kcal/mol)', 'Derivatives'])
    return fes_df


def fourier_series(x, f, n):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


def fourier_approximation(xdata, ydata, N=10):
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, w, N)}
    # Define a Fit object for this model and data
    fit = Fit(model_dict, x=xdata, y=ydata)
    fit_result = fit.execute()
    return fit_result, fit  # fit.model(x=xd, **fit_result.params).y


def mean_funtion(list_of_approximations, number_of_replicas):
    y = np.empty((number_of_replicas, list_of_approximations[0].shape[0]))
    print(y.shape)
    for i in range(0, number_of_replicas):
        fes = list_of_approximations[i]
        x = np.array(fes.iloc[:, 1].values, dtype=np.float32)
        y[i] = x
        # print(x)
    return np.average(y, axis=0)


def DFE(upper_limit, lower_limit):
    return -boltzman_constant * temp * log(romberg(Q, lower_limit, upper_limit) / (upper_limit - lower_limit))


# x is the CV
def gety(x, params):
    return params['a0'] + params['a1'] * cos(params['w'] * x) + params['a10'] * cos(10 * params['w'] * x) + params[
        'a2'] * cos(2 * params['w'] * x) + params['a3'] * cos(3 * params['w'] * x) + params['a4'] * cos(
        4 * params['w'] * x) + params['a5'] * cos(5 * params['w'] * x) + params['a6'] * cos(6 * params['w'] * x) + \
           params['a7'] * cos(7 * params['w'] * x) + params['a8'] * cos(8 * params['w'] * x) + params['a9'] * cos(
        9 * params['w'] * x) + params['b1'] * sin(params['w'] * x) + params['b10'] * sin(10 * params['w'] * x) + params[
               'b2'] * sin(2 * params['w'] * x) + params['b3'] * sin(3 * params['w'] * x) + params['b4'] * sin(
        4 * params['w'] * x) + params['b5'] * sin(5 * params['w'] * x) + params['b6'] * sin(6 * params['w'] * x) + \
           params['b7'] * sin(7 * params['w'] * x) + params['b8'] * sin(8 * params['w'] * x) + params['b9'] * sin(
        9 * params['w'] * x)


def Q(x):  # nominal partition function
    return exp(-gety(x, params) / (boltzman_constant * temp))


def getMinMaxCV(FES_data_list):
    min = []
    max = []
    m = 1
    for df in FES_data_list:
        if df.shape[0] > m:
            m = df.shape[0]
        max.append(df.max()[0])
        min.append(df.min()[0])
        # # print(df)
        # print()
    min.sort()
    max.sort()
    #
    return min[0], max[0], np.linspace(min[0], max[0], num=m)


def main():
    # test()
    # Get all replica directories
    root = '/gpfs/gsfs1/scr/gsabbih/Isda/Fold&Dock/10/gromacs/gromacs'
    # Calculate Free energies from simulation for each replica
    fes_list = []
    for x in os.walk(root):
        if x[0].__contains__('/replica'):
            fes = calculateFES(x[0])
            fes['Replica name'] = x[0].split('/')[-1]
            # print(fes.head())
            fes_list.append(fes)

    # os.chdir(root)
    data = getMinMaxCV(fes_list)  # min and min of max distance of all replicas
    global params
    l = []
    count = 0
    # find an approximation of the all free energies with a fourier  approximation
    l1 = []
    for index, fes in enumerate(fes_list):
        fit_result, fit = fourier_approximation(fes.iloc[:, 0].values, fes.iloc[:, 1].values,N=10)
        dist = np.linspace(fes.min()[0], fes.max()[0], num=fes.shape[0])
        l.append(pd.DataFrame({'dist': data[2], 'FES': fit.model(x=data[2], **fit_result.params).y}))
        fes['Fit FES(kcal/mol)'] = fit.model(x=dist, **fit_result.params).y
        l1.append(fes)
        count += 1
    fes_df = pd.concat(l1)
    fes_df.to_csv(root + "/FES_Apt_{}.csv".format(root.split('/')[-3]))
    print(fes_df.head())
    convx = []
    convDFE = []
    res_mean_fes = []
    print(len(l))
    for i in range(0, len(l)):
        y = mean_funtion(l, i + 1)
        # print('mean function: ', y)
        fit_result, fit  = fourier_approximation(data[2], y)
        df = pd.DataFrame({'Dist(nm)': data[2], 'FES(kcal/mol)': y, 'Fit FES(kcal/mol)': fit.model(x=data[2], **fit_result.params).y})
        df['Windows(N)'] = i + 1
        res_mean_fes.append(df)
        params = dict(fit_result.params)
        DFE = -boltzman_constant * temp * log(romberg(Q, data[0], data[1], divmax=20) / (data[1] - data[0]))
        convx.append(i + 1)
        convDFE.append(DFE)

    dfe = pd.DataFrame({"Windows(N)": convx, "DFE(kcal/mol)": convDFE})
    res_mean_fes = pd.concat(res_mean_fes)
    res_mean_fes.to_csv((root + "/win_FES_Apt_{}.csv".format(root.split('/')[-3])))
    dfe.to_csv((root + "/DFE_Apt_{}.csv".format(root.split('/')[-3])))


# u = mda.Universe("/gpfs/gsfs1/scr/gsabbih/Isda/Fold&Dock/1/MDS/metaD_gromacs/step3_input.pdb")
# pro = ','.join(str(e.ix+1) for e in u.select_atoms('protein and backbone'))
# nuc = ','.join(str(e.ix+1) for e in u.select_atoms('nucleicbackbone'))
#
# prepare_plumed_config('step3_input.pdb', pro, nuc)
# prep_replicas(28)


def f(x):
    return gety(x)


if __name__ == '__main__':
    main()

