import os, sys,time, argparse
from datetime import date
import random
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# import cleanfid


from cleanfid.utils import *
from cleanfid.features import *
from cleanfid.resize import *


@tf.function
def D(X,C,order):
    #####
    ## X :      Data of dimension N x n - batch size N and dimensionality n
    ## C :      Centers from data/generator of dimension M x n - M is batch size
    ## D(x) :   Computes the "Columb Potential" of each x_i, given all c_i of dimension Nx1
    ##
    #####
    ### Data vector
    N = X.shape[0] #### Batch size of data
    M = C.shape[0] #### Batch size of each set of centers

    W = (1/N)*tf.ones([C.shape[0]])
    W = tf.expand_dims(W, axis = 1)
    # print('W', W)

    X = tf.expand_dims(X, axis = 2) ## Nxnx1
    X = tf.expand_dims(X, axis = 1) ## Nx1xnx1
    # print('Input X',X, X.shape)

    C = tf.expand_dims(C, axis = 2) ## Mxnx1
    C = tf.expand_dims(C, axis = 0) ## 1xMxnx1
    # print('Centers C', C, C.shape)

    C_tiled = tf.tile(C, [N,1,1,1])  ## NxMxnx1 ### was tf.shape(X)[0]
    X_tiled = tf.tile(X, [1,M,1,1])  ## NxMxnx1 ### was self.num_hidden
    # print('C_tiled', C_tiled, C_tiled.shape)
    # print('X_tiled', X_tiled, X_tiled.shape)

    Tau = C_tiled - X_tiled ## Nx2Mxnx1 = Nx2Mxnx1 - Nx2Mxnx1
    # print('Tau', Tau)

    #### Columb power is ||x||^{-3}? --- check

    # order = order
    sign = 1.
    if order < 0:
        sign *= -1

    norm_tau = tf.norm(Tau, ord = 'euclidean', axis = 2)
    ord_tensor = order*tf.ones_like(norm_tau)
    Phi = sign*tf.pow(norm_tau, ord_tensor) ## Nx1

    W = tf.cast(W, dtype = 'float32')
    Phi = tf.cast(Phi, dtype = 'float32')
    D_val = tf.squeeze(tf.linalg.matmul(W, Phi, transpose_a=True, transpose_b=False),axis = 2)

    return D_val

# @tf.function
def SID(D_g,D_d):
    return tf.reduce_mean(D_g) - tf.reduce_mean(D_d)


"""
Compute the sid score given the sets of features
"""
def signed_distance(feats1, feats2, order, num_subsets=20, max_subset_size=100):


    ### in case I need it...
    n = feats1.shape[1]
    Unif_batch_size = m = 300#min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)

    range_var = range(1,200)
    r_scaling = 0.5
    sid_vec = np.zeros([len(range_var),2])

    mu_d = np.mean(feats2,axis = 0)
    cov_d = np.cov(feats2,rowvar = False)
    step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))

    # for r in range_var:
    for r in tqdm(range_var, desc='SID vs. r'):
        # print(r)
        sid_vec[r-1,1] = r_scaling*(r-1)
        f2 = feats2.copy()
        f1 = feats1.copy()
        for _subset_idx in range(num_subsets):
            # print(_subset_idx)

            ### act2 = feats of reals
            act2 = f2[np.random.choice(f2.shape[0], m, replace=False)]
            ### act1 = feats of fakes
            act1 = f1[np.random.choice(f1.shape[0], m, replace=False)]      

            cur_step = tf.reduce_mean(r_scaling*(r/2)*step_size)         

            # print(act2,act1,cur_step) 

            Unif = tfp.distributions.Uniform(low=mu_d - cur_step, high=mu_d + cur_step)
            X = tf.cast(Unif.sample([Unif_batch_size]),dtype = 'float32')

            D_d = D(X,act2,order)
            D_g = D(X,act1,order)

            # print(D_d,D_g)

            cur_sid_value = SID(D_g,D_d)
            # print(cur_sid_value)

            if sid_vec[r-1,0] == 0:
                sid_vec[r-1,0] = cur_sid_value
            else:
                sid_vec[r-1,0] = 0.5*sid_vec[r-1,0] + 0.5*cur_sid_value

        # print(sid_vec)
    return sid_vec

  
"""
Compute the inception features for a batch of images
"""
def get_batch_features(batch, model, device):
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()


"""
Compute the inception features for a list of files
"""
def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None, 
                       description=""):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, mode=mode)
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    for batch in tqdm(dataloader, desc=description):
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a folder of image files
"""
def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description=""):
    # get all relevant files in the dataset
    files = sorted([file for ext in EXTENSIONS
                    for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)])
    print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device,
                                  mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  description=description)
    return np_feats



"""
Compute the FID stats from a generator model
"""
def get_model_features(G, model, mode="clean", z_dim=512, 
        num_gen=50_000, batch_size=128,
        device=torch.device("cuda"), desc="FID model: "):
    fn_resize = build_resizer(mode)
    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    for idx in tqdm(range(num_iters), desc=desc):
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                resized_batch = torch.zeros(batch_size, 3, 299, 299)
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    img_resize = fn_resize(img_np)
                    resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)))
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute SID using the clean-fid codebase
"""
def compute_sid(fdir1=None, fdir2=None, order = 1, gen=None, 
            mode="clean", num_workers=12, batch_size=32,
            device=torch.device("cuda"), dataset_name="FFHQ",
            dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    #### It is an assymetrical cost. Please note!!!!!!!!!!!!!!!!!
    #### fdir1 = Fakes; fdir2 = Reals
    
    # if both dirs are specified, compute KID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute SID between two folders")
        # get all inception features for the first folder
        fbname1 = os.path.basename(fdir1)
        np_feats1 = get_folder_features(fdir1, None, num_workers=num_workers,
                            batch_size=batch_size, device=device, mode=mode, 
                            description=f"SID {fbname1} : ")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(fdir2, None, num_workers=num_workers,
                            batch_size=batch_size, device=device, mode=mode, 
                            description=f"SID {fbname2} : ")
        score_vec = signed_distance(np_feats1, np_feats2, order)
        return score_vec

    # compute sid of a folder with stored statistics
    # Should not work for now. SID supports only folders
    elif fdir1 is None or fdir2 is None:

        print(f"computing SID of a folder with {dataset_name} statistics not supported. ")
        exit(0)


    #     print(f"compute KID of a folder with {dataset_name} statistics")
    #     # define the model if it is not specified
    #     model = build_feature_extractor(mode, device)
    #     ref_feats = get_reference_statistics(dataset_name, dataset_res,
    #                         mode=mode, seed=0, split=dataset_split, metric="KID")
    #     fbname = os.path.basename(fdir1)
    #     # get all inception features for folder images
    #     np_feats = get_folder_features(fdir1, model, num_workers=num_workers,
    #                                     batch_size=batch_size, device=device,
    #                                     mode=mode, description=f"KID {fbname} : ")
    #     score_vec = signed_distance(ref_feats, np_feats)
    #     return score_vec

    # # compute fid for a generator
    # elif gen is not None:
    #     print(f"compute KID of a model with {dataset_name}-{dataset_res} statistics")
    #     # define the model if it is not specified
    #     model = build_feature_extractor(mode, device)
    #     ref_feats = get_reference_statistics(dataset_name, dataset_res,
    #                         mode=mode, seed=0, split=dataset_split, metric="KID")
    #     # build resizing function based on options
    #     fn_resize = build_resizer(mode)
    #     # Generate test features
    #     np_feats = get_model_features(gen, model, mode=mode,
    #         z_dim=z_dim, num_gen=num_gen, desc="KID model: ",
    #         batch_size=batch_size, device=device)
    #     score_vec = signed_distance(ref_feats, np_feats)
    #     return score_vec
    
    # else:
    #     raise ValueError(f"invalid combination of directories and models entered")


def print_and_save_sid(path ='logs/', SID_vec = None):

    from matplotlib.backends.backend_pgf import PdfPages
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "helvetica",  # use serif/main font for text elements
        "font.size":12,
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    })

    max_list = []
    min_list = []
    with PdfPages(path+'_plot.pdf') as pdf:
        # for SID_vals,locs in SID_vec:
        fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
        ax1 = fig1.add_subplot(111)
        ax1.cla()
        ax1.get_xaxis().set_visible(True)
        ax1.get_yaxis().set_visible(True)
        SID_vals = SID_vec[:,0]
        ymax = max(SID_vals)
        ymin = min(SID_vals)
        locs = SID_vec[:,1]
        ax1.plot(locs, SID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
        ax1.set_xlabel(r'$r:\,\mathcal{U}\left[\mu_d - r \sigma_d, \mu_d + r \sigma_d\right] $')
        ax1.set_ylabel(r'$SID(p_g|p_d)$')
        title = 'SID of the Final Model'
        plt.title(title, fontsize=8)
        pdf.savefig(fig1)
        plt.close(fig1)

    # ymin = min(min_list)
    # ymax = max(max_list)
    with PdfPages(path+'_MinMaxY_plot.pdf') as pdf:
        # for SID_vals,locs in SID_vec:
        fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
        ax1 = fig1.add_subplot(111)
        ax1.cla()
        ax1.set_ylim([0.95*ymin, 1.05*ymax])
        ax1.get_xaxis().set_visible(True)
        ax1.get_yaxis().set_visible(True)
        SID_vals = SID_vec[:,0]
        locs = SID_vec[:,1]
        ax1.plot(locs, SID_vals, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
        ax1.set_xlabel(r'$r:\,\mathcal{U}\left[\mu_d - 0.1 r \sigma_d, \mu_d + 0.1 r \sigma_d\right] $')
        ax1.set_ylabel(r'$SID(p_g|p_d)$')
        title = 'SID of the Final Model'
        plt.title(title, fontsize=8)
        pdf.savefig(fig1)#, bbox_inches='tight', dpi=400)
        plt.close(fig1)

    # with PdfPages(path+'_LogY_plot.pdf') as pdf:
    #     # for SID_vals,locs in SID_vec:
    #     fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
    #     ax1 = fig1.add_subplot(111)
    #     ax1.cla()
    #     ax1.set_ylim([ymin, ymax])
    #     ax1.get_xaxis().set_visible(True)
    #     ax1.get_yaxis().set_visible(True)
    #     SID_vals = SID_vec[:,0]
    #     locs = SID_vec[:,1]
    #     ax1.plot(locs, np.log10(np.abs(SID_vals+1e-10)), color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
    #     ax1.set_xlabel(r'$r:\,\mathcal{U}\left[\mu_d - 0.1 r \sigma_d, \mu_d + 0.1 r \sigma_d\right] $')
    #     ax1.set_ylabel(r'$\log\left(SID(p_g|p_d)\right)$')
    #     title = 'SID of the Final Model'
    #     plt.title(title, fontsize=8)
    #     pdf.savefig(fig1)#, bbox_inches='tight', dpi=400)
    #     plt.close(fig1)

    today = date.today()
    d1 = today.strftime("%d%m%Y")
    np.save(path+'_Vec'+d1+'.npy',np.array(SID_vec))


# if __name__ == '__main__':

 
#     #### It is an assymetrical cost. Please note!!!!!!!!!!!!!!!!!
#     #### fdir1 = Fakes; fdir2 = Reals
#     sid_vec = compute_sid(fdir1 = FLAGS_dict['fakes_dir'], fdir2 = FLAGS_dict['reals_dir'], o = FLAGS_dict)

#     print_and_save_sid(plot_path, sid_vec)


