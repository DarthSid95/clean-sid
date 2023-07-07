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
from cleanfid import fid


from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('data', 'mnist', """Type of Data to run for""")
flags.DEFINE_string('model', 'Base', """Type of Data to run for""")
flags.DEFINE_string('noise', 'gaussian', """Type of Data to feed as noise""")
flags.DEFINE_string('log_folder', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('run_id', 'new', """ID of the run, used in saving.""")
flags.DEFINE_string('reals_dir', None, """Location of real images.""")
flags.DEFINE_string('fakes_dir', None, """Location of fake images.""")
flags.DEFINE_integer('resume', 0, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('FID_flag', 0, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('batch_size', 100, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_string('metric_mode', 'clean', """ clean/legacy_tensorflow/legacy_pytorch""")
flags.DEFINE_integer('KID_flag', 0, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('SID_flag', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('order', 1, """Order integer n """)
flags.DEFINE_string('GPU', '0,1', """GPU's made visible '0', '1', or '0,1' """)





"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""
# def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)
#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert mu1.shape == mu2.shape, \
#         'Training and test mean vectors have different lengths'
#     assert sigma1.shape == sigma2.shape, \
#         'Training and test covariances have different dimensions'

#     diff = mu1 - mu2

#     # Product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = ('fid calculation produces singular product; '
#                'adding %s to diagonal of cov estimates') % eps
#         print(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError('Imaginary component {}'.format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

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
    return tf.reduce_sum(D_g) - tf.reduce_sum(D_d)
    # return tf.reduce_mean(D_g) - tf.reduce_mean(D_d)


"""
Compute the sid score given the sets of features
"""
def signed_distance(feats1, feats2, FLAGS_dict, num_subsets=20, max_subset_size=100):

    ### My SID geenralized code here; now.

    ### in case I need it...
    n = feats1.shape[1]
    Unif_batch_size = m = 500#min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)

    range_var = range(1,50) ## 200 for m1; 500 for 1
    r_scaling = 0.5
    sid_vec = np.zeros([len(range_var),2])

    numImgs = min(feats1.shape[0],feats2.shape[0])

    mu_d = np.mean(feats2,axis = 0)
    cov_d = np.cov(feats2,rowvar = False)
    step_size = tf.math.sqrt(tf.linalg.diag_part(cov_d))

    # for r in range_var:
    for r in tqdm(range_var, desc='SID vs. r'):
        # print(r)
        sid_vec[r-1,1] = r_scaling*(r-1)
        f2 = feats2.copy()[0:numImgs]
        f1 = feats1.copy()[0:numImgs]
        # print(f2.shape,f1.shape)
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

            D_d = D(X,act2,FLAGS_dict['order'])
            D_g = D(X,act1,FLAGS_dict['order'])

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
                       batch_size=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), desc="FID model: "):
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
def compute_sid(fdir1=None, fdir2=None, FLAGS_dict = None, gen=None, 
            mode="clean", num_workers=12, batch_size=32,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dataset_name="FFHQ",
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
        score_vec = signed_distance(np_feats1, np_feats2, FLAGS_dict)
        return score_vec

    # compute sid of a folder with stored statistics
    # Should not work for now. SID supports only folders
    elif fdir1 is not None and fdir2 is None:

        print(f"computing SID of a folder with {dataset_name} statistics not supported. ")
        exit(0)


        print(f"compute KID of a folder with {dataset_name} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, seed=0, split=dataset_split, metric="KID")
        fbname = os.path.basename(fdir1)
        # get all inception features for folder images
        np_feats = get_folder_features(fdir1, model, num_workers=num_workers,
                                        batch_size=batch_size, device=device,
                                        mode=mode, description=f"KID {fbname} : ")
        score_vec = signed_distance(ref_feats, np_feats)
        return score_vec

    # compute fid for a generator
    elif gen is not None:
        print(f"compute KID of a model with {dataset_name}-{dataset_res} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, seed=0, split=dataset_split, metric="KID")
        # build resizing function based on options
        fn_resize = build_resizer(mode)
        # Generate test features
        np_feats = get_model_features(gen, model, mode=mode,
            z_dim=z_dim, num_gen=num_gen, desc="KID model: ",
            batch_size=batch_size, device=device)
        score_vec = signed_distance(ref_feats, np_feats)
        return score_vec
    
    else:
        raise ValueError(f"invalid combination of directories and models entered")


def print_and_save_sid(path ='logs/',res_file = 'Results.txt', SID_vec = None):

    from matplotlib.backends.backend_pgf import PdfPages
    # plt.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     "font.family": "helvetica",  # use serif/main font for text elements
    #     "font.size":12,
    #     "text.usetex": True,     # use inline math for ticks
    #     "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    # })

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
    # for SID_vals,locs in SID_vec:
    #     continue

    today = date.today()
    d1 = today.strftime("%d%m%Y")
    np.save(path+'_Vec'+d1+'.npy',np.array(SID_vec))

    

    # p001_r = locs[np.where(SID_vals <= 0.001)][0]
    sum_sid = np.sum(SID_vals)

    
    f = open(res_file,'a')
    # f.write("\n r to acieve 10% of SID_min :"+str(p001_r)+'\n')
    f.write("\n Sum SID :"+str(sum_sid)+'\n')
    f.close()
    print('\n Sum SID :'+str(sum_sid)+'\n')


FLAGS(sys.argv)
if __name__ == '__main__':

    FLAGS_dict = FLAGS.flag_values_dict()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS_dict['GPU']

    if FLAGS_dict['log_folder'] == 'default':
        today = date.today()
        log_dir = 'logs/Log_Folder_'+today.strftime("%d%m%Y")+'/'
    else:
        log_dir = FLAGS_dict['log_folder']
    
    if log_dir[-1] != '/':
        log_dir += '/' 

    if FLAGS_dict['resume']:     
        run_loc = log_dir + FLAGS_dict['run_id']
        print("Resuming from folder {}".format(run_loc))
    else:
        print("No RunID specified. Logs will be saved in a folder based on FLAGS")  
        today = date.today()
        d1 = today.strftime("%d%m%Y")
        run_id = d1 +'_sid_data_' + FLAGS_dict['data']+ '_noise_' + FLAGS_dict['noise'] + '_' + FLAGS_dict['model'] + '_order_' + str(FLAGS_dict['order']).replace('-','m')
        # self.run_id = d1 +'_'+ self.topic + '_' + self.data + '_' + self.gan + '_' + self.loss
        run_loc = log_dir + run_id

        runs = sorted(glob(run_loc+'*/'))
        print(runs)
        if len(runs) == 0:
            curnum = 0
        else:
            curnum = int(runs[-1].split('_')[-1].split('/')[0])
        print(curnum)
        if FLAGS_dict['run_id'] == 'new':
            curnum = curnum+1
        else:
            curnum = curnum
            if FLAGS_dict['run_id'] != 'same' and os.path.exists(run_loc + '_' + str(curnum).zfill(2)):
                x = input("You will be OVERWRITING existing DATA. ENTER to continue, type N to create new ")
                if x in ['N','n']:
                    curnum += 1
        run_loc += '_'+str(curnum).zfill(2)


    ''' Create for/ Create base logs folder'''
    pwd = os.popen('pwd').read().strip('\n')
    if not os.path.exists(pwd+'/logs'):
        os.mkdir(pwd+'/logs')

    ''' Create log folder / Check for existing log folder'''
    if os.path.exists(log_dir):
        print("Directory " , log_dir ,  " already exists")
    else:
        os.mkdir(log_dir)
        print("Directory " , log_dir ,  " Created ")


    if os.path.exists(run_loc):
        print("Directory " , run_loc ,  " already exists")
    else:   
        if FLAGS_dict['resume']:
            print("Cannot resume. Specified log does not exist")
        else:   
            os.mkdir(run_loc)
            print("Directory " , run_loc ,  " Created ") 


    plot_path = run_loc + '/sid_data_' + FLAGS_dict['data']+ '_noise_' + FLAGS_dict['noise'] + '_' + FLAGS_dict['model'] + '_order_' + str(FLAGS_dict['order'])

    res_file = plot_path+'_Results.txt'

    FLAGS.append_flags_into_file(plot_path+'_Flags.txt')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    #### It is an assymetrical cost. Please note!!!!!!!!!!!!!!!!!
    #### fdir1 = Fakes; fdir2 = Reals
    if FLAGS_dict['SID_flag']:
        sid_vec = compute_sid(fdir1 = FLAGS_dict['fakes_dir'], fdir2 = FLAGS_dict['reals_dir'], FLAGS_dict = FLAGS_dict)
        print_and_save_sid(plot_path, res_file, sid_vec)

    if FLAGS_dict['FID_flag']:
        fid_val = fid.compute_fid(FLAGS_dict['fakes_dir'], FLAGS_dict['reals_dir'], mode = FLAGS_dict['metric_mode'],device=device, batch_size = FLAGS_dict['batch_size'])
        f = open(res_file,'a')
        f.write("\n FID :"+str(fid_val)+'\n')
        f.close()
        print('\n FID :'+str(fid_val)+'\n')
    if FLAGS_dict['KID_flag']:
        kid_val = fid.compute_kid(FLAGS_dict['fakes_dir'], FLAGS_dict['reals_dir'], mode = FLAGS_dict['metric_mode'],device=device, batch_size = FLAGS_dict['batch_size'])
        f = open(res_file,'a')
        f.write("\n KID :"+str(kid_val)+'\n')
        f.close()
        print('\n KID :'+str(kid_val)+'\n')

    

