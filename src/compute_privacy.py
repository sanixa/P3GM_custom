import hivae
import vae
import sys
import torch
import functools
import my_util
import math
import pathlib
import argparse
import numpy as np
filedir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(filedir.parent / "privacy"))
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

sys.path.append(str(filedir.parent))
import dp_utils


# The method to compute the sum of the privacy budget for P3GM.
# This method is depending on the tensorflow.privacy library
def analysis_privacy(lot_size, data_size, sgd_sigma, gmm_sigma, gmm_iter, gmm_n_comp, sgd_epoch, pca_eps, delta=1e-5):
    q = lot_size / data_size
    sgd_steps = int(math.ceil(sgd_epoch * data_size / lot_size))
    gmm_steps = gmm_iter * (2 * gmm_n_comp + 1)
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
    pca_rdp = np.array(orders) * 2 * (pca_eps**2)
    sgd_rdp = compute_rdp(q, sgd_sigma, sgd_steps, orders)
    gmm_rdp = compute_rdp(1, gmm_sigma, gmm_steps, orders)
    
    rdp = pca_rdp + gmm_rdp + sgd_rdp
    
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    
    index = orders.index(opt_order)
    print(f"ratio(pca:gmm:sgd):{pca_rdp[index]/rdp[index]}:{gmm_rdp[index]/rdp[index]}:{sgd_rdp[index]/rdp[index]}")
    print(f"GMM + SGD + PCA (MA): {eps}, {delta}-DP")
    
    return eps, [pca_rdp[index]/rdp[index], gmm_rdp[index]/rdp[index], sgd_rdp[index]/rdp[index]]


# The method to construct the P3GM class.
# We prepare two types of P3GM which are depending on VAE and HI-VAE.
# Refer to https://arxiv.org/abs/1807.03653 for HI-VAE.
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_sigma', '-n', type=float, default=None, help='noise_sigma')
    parser.add_argument('--epoch', '-e', type=int, default=None, help='epoch')
    args = parser.parse_args()



    # compute the sum of privacy budgets using RDP
    print(analysis_privacy(240, 62880, args.noise_sigma, 120.0, 20, 1,  args.epoch, 0.01, delta=1e-5))
        
if __name__ == '__main__':
    main()
