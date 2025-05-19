import argparse
import copy
from ctypes import c_int,c_double
import re
import os
import multiprocessing
import sys

import click
import pandas
import hydropt.hydropt as hd
import numpy as np
from matplotlib import pyplot as plt
import lmfit
import pandas
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from spectral import envi

from hydropt.bio_optics import H2O_IOP_DEFAULT
from hydropt.bio_optics import a_phyto_base_full
from hydropt.bio_optics import nap 
from hydropt.utils import interpolate_to_wavebands
from hydropt.hydropt import PolynomialReflectance

from isofit.core.common import envi_header
from isofit import ray
from isofit.core.fileio import IO, write_bil_chunk


def clear_nat_water(*args, wb):
    H2O_IOP = interpolate_to_wavebands(
        H2O_IOP_DEFAULT,
        wavelength=wb
    )
    return H2O_IOP.T.values


def phytoplankton(*args, wb):
    chl = args[0]
    # basis vector - according to Ciotti&Cullen (2002)

    a_phyto_base = interpolate_to_wavebands(
        data=a_phyto_base_full, 
        wavelength=wb
    )
    a = 0.06 * chl * a_phyto_base.absorption.values
    # constant spectral backscatter with backscatter ratio of 1.4%
    bb = np.repeat(.014 * 0.18 * chl, len(a))

    return np.array([a, bb])


def cdom(*args, wb):
    # absorption at 440 nm
    a_440 = args[0]

    # spectral absorption
    a = np.array(np.exp(-0.017 * (wb - 440)))

    # no backscatter
    bb = np.zeros(len(a))

    return a_440*np.array([a, bb])


def nap(*args, wb):
    '''
    IOP model for NAP
    '''
    spm = args[0]
    # slope = args[1]


    # Absoprtion
    a = (.041 * .75 * np.exp(-.0123 * (wb - 443)))

    slope = 0.14 * 0.57
    bb = slope * (550 / wb)
    
    return spm * np.array([a, bb]) 



def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def fill_helper(y, val):

    return y == val, lambda z: z.nonzero()[0]

def fit_wq(rrs, inv_model, x0, fwd_model, fm, wl_use):
    # Find glint:
    # rrs = (rrs.copy() - np.mean(rrs[glint_idx])) / np.pi
    rrs = rrs.copy() / np.pi
    nans, x = nan_helper(rrs)
    fill, x = fill_helper(rrs, -.01)
    rrs[nans]= np.interp(x(nans), x(~nans), rrs[~nans])
    rrs[fill]= np.interp(x(fill), x(~fill), rrs[~fill])
    rrs = rrs[wl_use].astype(float)

    xhat = inv_model.invert(y=rrs, x=x0)

    res = vars(xhat).get('uvars')
    params = xhat.params

    if not params:
        phyto_conc = -9999
        phyto_conc_std = -9999

        cdom_conc = -9999
        cdom_conc_std = -9999

        nap_conc = -9999
        nap_conc_std = -9999


    elif res:
        phyto_conc = xhat.uvars['phyto'].nominal_value
        phyto_conc_std = xhat.uvars['phyto'].std_dev

        cdom_conc = xhat.uvars['cdom'].nominal_value
        cdom_conc_std = xhat.uvars['cdom'].std_dev

        nap_conc = xhat.uvars['nap'].nominal_value
        nap_conc_std = xhat.uvars['nap'].std_dev

    else:
        phyto_conc = float(xhat.params['phyto'].value)
        phyto_conc_std = -9999

        cdom_conc = float(xhat.params['cdom'].value)
        cdom_conc_std = -9999

        nap_conc = float(xhat.params['nap'].value)
        nap_conc_std = -9999
    
    out_conc = [phyto_conc, cdom_conc, nap_conc]
    out_std = [phyto_conc_std, cdom_conc_std, nap_conc_std]

    iops = fm.iop_model.sum_iop(**{
        'phyto': phyto_conc,
        'cdom': cdom_conc,
        'nap': nap_conc
    })
    fwd = fm.refl_model.forward(iops)

    return out_conc, out_std, fwd


@ray.remote(num_cpus=1)
def run_chunk(
    data,
    x0, 
    inv_model,
    fwd_model,
    fm,
    lstart, 
    lend, 
    wavebands,
    wl_use,
):
    
    chunk = data[lstart:lend, ...]

    concs = np.zeros((len(chunk), 3))
    stds = np.zeros((len(chunk), 3))
    fwds = np.zeros((len(chunk), len(wavebands)))
    for i, sp in enumerate(chunk):
        print(i)
        conc, std, fwd = fit_wq(
            sp, 
            inv_model, 
            x0,
            fwd_model,
            fm,
            wl_use
        )
        concs[i, :] = conc
        stds[i, :] = std
        fwds[i, :] = fwd

    return concs, stds, fwds
    

@click.command()
@click.argument("path")
@click.argument("out_root")
@click.option("--n_cores", type=int, default=-1)
def main(
    path,
    out_root,
    n_cores: int = 1,
    ray_address: str = None,
    ray_redis_password: str = None,
    ray_temp_dir=None,
    ray_ip_head=None,
):

    path = '/Users/bgreenbe/Projects/SWOT/Mississippi1/Data/CSV/EMIT_L2A_RFL_001_20241002T165450_sample.csv'
    out_root = '/Users/bgreenbe/Projects/SWOT/Mississippi1/Data/CSV'
    df = pandas.read_csv(path)
    wl = df['wl'].values
    wl_use = ((wl >= 400) & (wl <= 710))
    wavebands = wl[wl_use]

    # glint_idx = glint_idx = np.where((wl > 2100) & (wl < 2400))[0]
    bio_opt = hd.BioOpticalModel()
    bio_opt.set_iop(
        wavebands=wavebands,
        water=clear_nat_water,
        phyto=phytoplankton,
        cdom=cdom,
        nap=nap,
    )
    fwd_model = hd.PolynomialForward(bio_opt)
    fm = hd.ForwardModel(bio_opt, PolynomialReflectance())

    # set initial guess parameters for LM
    x0 = lmfit.Parameters()
    x0.add('phyto', value=1, min=1e-9)
    x0.add('cdom', value=1, min=1e-9)
    x0.add('nap', value=1, min=1e-9)

    inv_model = hd.InversionModel(
        fwd_model=fwd_model,
        minimizer=lmfit.minimize
    )
    
    # Start up a ray instance for parallel work
    rayargs = {
        "ignore_reinit_error": True,
        "local_mode": n_cores == 1,
        "address": ray_address,
        "include_dashboard": False,
        "_temp_dir": ray_temp_dir,
        "_redis_password": ray_redis_password,
    }

    # We can only set the num_cpus if running on a single-node
    if ray_ip_head is None and ray_redis_password is None:
        rayargs["num_cpus"] = n_cores

    ray.init(**rayargs)

    # Construct array of spectra
    cols = df.columns[2:]
    data = df[cols].to_numpy().T
    nl = len(data)

    jobs = []
    nchunk = n_cores
    step = (nl // nchunk) + 1
    range = np.arange(0, nl, step)
    for lstart in range:
        lend = min(lstart + step, nl)
        if lend > (nl):
            lend = nl

        jobs.append(
            run_chunk.remote(
                data,
                x0, 
                inv_model,
                fwd_model,
                fm,
                lstart, 
                lend, 
                wavebands,
                wl_use,
            )
        )

    rreturn = [ray.get(jid) for jid in jobs]

    concs = np.zeros((len(data), 3))
    stds = np.zeros((len(data), 3))
    fwds = np.zeros((len(data), len(wavebands)))
    # Accumulate returns
    i = 0
    for r in rreturn:
        print(i)
        conc = r[0]
        std = r[1]
        fwd = r[2]

        concs[i:i+len(r[0]), :] = r[0]
        stds[i:i+len(r[1]), :] = r[1]
        fwds[i:i+len(r[2]), :] = r[2]
        i += len(r[0])

    conc_df = pandas.DataFrame(concs, columns=['Phyto', 'Cdom', 'Nap'])
    std_df = pandas.DataFrame(stds, columns=['Phyto', 'Cdom', 'Nap'])
    fwds_df = pandas.DataFrame(fwds, columns=wavebands).transpose().reset_index(
        drop=False, names=['wavelength']
    )
    # Save raw data
    match = re.search('(EMIT_L2A_RFL_.{3}_\d{8}T\d{6})', path)
    if match:
        out_name = match.groups(1)[0]
    else:
        raise ValueError('Could not match file id. Check.')
    out_path = os.path.join(out_root, out_name + '_concentrations.csv')
    conc_df.to_csv(out_path)
    out_path = os.path.join(out_root, out_name + '_std.csv')
    std_df.to_csv(out_path)
    out_path = os.path.join(out_root, out_name + '_forward.csv')
    fwds_df.to_csv(out_path)

    # Get stats
    conc_means = np.nanmean(concs, axis=0)
    conc_stds = np.nanstd(concs, axis=0)
    conc_median = np.nanmedian(concs, axis=0)
    conc_min = np.nanmin(concs, axis=0)
    conc_max = np.nanmax(concs, axis=0)

    summary_df = pandas.DataFrame(data={
        'stat': ['mean', 'std', 'median', 'min', 'max'],
        'phyto': [
            conc_means[0],
            conc_stds[0],
            conc_median[0],
            conc_min[0],
            conc_max[0],
        ],
        'cdom': [
            conc_means[1],
            conc_stds[1],
            conc_median[1],
            conc_min[1],
            conc_max[1],
        ],
        'nap': [
            conc_means[2],
            conc_stds[2],
            conc_median[2],
            conc_min[2],
            conc_max[2],
        ],
    })
    out_path = os.path.join(out_root, out_name + '_wq_summary.csv')
    summary_df.to_csv(out_path)


if __name__ == '__main__':
    main()
