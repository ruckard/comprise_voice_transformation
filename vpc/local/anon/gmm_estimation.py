'''
This is a script to measure GMM density
of x-vectors in high dimension.
It needs spk2gender and spk_xvector.scp
'''
import sys


import os
from os.path import join, basename, isfile
import random

import numpy as np
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.externals import joblib

from kaldiio import WriteHelper, ReadHelper

args = sys.argv
print(args)

src_data_dir = args[1]
pool_data_dir = args[2]
xvec_out_dir = args[3]
pseudo_xvec_dir = args[4]
rand_level = args[5]
cross_gender = args[6] == "true"
proximity = args[7]
rand_seed = int(args[8])


random.seed(rand_seed)
N = 32

def estimate_density(pool_xvecs, pool_gender, N, gmm_dir):
    gmm_file =  join(gmm_dir, 'gmm.pkl')
    meta_file = join(gmm_dir, 'meta.txt')
    mean_file = join(gmm_dir, 'mean.npy')
    std_file = join(gmm_dir, 'std.npy')
    if not isfile(gmm_file):
        # Normalize x-vectors
        mean_Xvecs = np.mean(pool_xvecs, axis=0)
        std_Xvecs = np.std(pool_xvecs, axis=0)
        pool_xvecs = (pool_xvecs - mean_Xvecs) / std_Xvecs
        
        # Learn GMM labels
        gmm = GaussianMixture(n_components=N, random_state=rand_seed, covariance_type='full')
        #gmm = BayesianGaussianMixture(n_components=N, random_state=rand_seed, covariance_type='full')
        labels = gmm.fit(pool_xvecs).predict(pool_xvecs)

        component_densities = np.empty_like(gmm.weights_)
        for i, (covar, w) in enumerate(zip(gmm.covariances_, gmm.weights_)):
            component_densities[i] = w / np.abs(np.linalg.det(covar))

        rank_weights = np.argsort(gmm.weights_)
        rank_densities = np.argsort(component_densities)

        # Save GMM and metadata
        os.makedirs(gmm_dir)
        _ = joblib.dump(gmm, gmm_file)
        # Save mean and std
        np.save(mean_file, mean_Xvecs)
        np.save(std_file, std_Xvecs)
        with open(meta_file, 'w') as f:
            f.write("component weight_rank density_rank gender\n")
            for i in range(N):
                c_pred = np.where(labels == i)[0]
                m_count = len([pool_gender[x] for x in c_pred.tolist() if pool_gender[x] == 'm'])
                f_count = len([pool_gender[x] for x in c_pred.tolist() if pool_gender[x] == 'f'])
                c_gender = 'm' if m_count > f_count else 'f'
                f.write(f"{str(i)} {rank_weights[i]} {rank_densities[i]} {c_gender}\n")
    
    # Load GMM and metadata
    gmm = joblib.load(gmm_file)
    gmm_meta = {}
    with open(meta_file) as f:
        for line in f.read().splitlines()[1:]:
            sp = line.split()
            gmm_meta[int(sp[0])] = (int(sp[1]), int(sp[2]), sp[3])
    mean_Xvecs = np.load(mean_file)
    std_Xvecs = np.load(std_file)
    return gmm, gmm_meta, mean_Xvecs, std_Xvecs

def znorm(xvecs, xmean, xstd):
    return (xvecs - xmean) / xstd

def predict_pseudo_xvecs(src_spks, src_spk2utt, src_xvecs, src_gender, gmm, gmm_meta, logfile):
    # Predict label for each source speaker
    log = open(logfile, 'w')
    src_labels = gmm.predict(src_xvecs)
    n_comp = len(gmm_meta.keys())

    gender_rev = {'m': 'f', 'f': 'm'}
    pseudo_xvec_map = {}
    pseudo_gender_map = {}
    pseudo_spk_xvecs = [] # t-SNE plotting
    # Sample from GMM based on constraints
    for spkid, label, gender in zip(src_spks, src_labels, src_gender):
        if cross_gender:
            gender = gender_rev[gender]
        # Change gender info
        pseudo_gender_map[spkid] = gender
        # Find all Gaussian components with required gender
        # But exclude the component to which this x-vector belong
        sel_gaussian_idx = [k for k, v in gmm_meta.items() 
                                if v[2] == gender and k != label]
        # Sample from dense or sparse regions
        # Find a single Gaussian component based on density rank
        sel_comp = sel_gaussian_idx[0]
        sel_comp_rank = gmm_meta[sel_comp][2]
        for idx in sel_gaussian_idx:
            if proximity == "dense":
                if gmm_meta[idx][2] > sel_comp_rank:
                    sel_comp = idx
                    sel_comp_rank = gmm_meta[idx][2]
            elif proximity == "sparse":
                if gmm_meta[idx][2] < sel_comp_rank:
                    sel_comp = idx
                    sel_comp_rank = gmm_meta[idx][2]

        # Create fake weights for GMM to select only one component
        gmm.weights_ = np.zeros((n_comp,))
        gmm.weights_[sel_comp] = 1.0
        log.write(f"Spkid: {spkid}, Original label: {label}, New label: {sel_comp}\n")

        # Sample from this Gaussian
        if rand_level == "spk":
            pseudo_xvec, nlab = gmm.sample()
            log.write(f"Changed from {label} to {nlab} during sampling\n\n")
            pseudo_spk_xvecs.append(pseudo_xvec)
            for uttid in src_spk2utt[spkid]:
                pseudo_xvec_map[uttid] = pseudo_xvec
        elif rand_level == "utt":
            for uttid in src_spk2utt[spkid]:
                pseudo_xvec, nlab = gmm.sample()
                pseudo_xvec_map[uttid] = pseudo_xvec
            pseudo_spk_xvecs.append(pseudo_xvec)
        else:
            print("rand_level not supported! Errors will happen!")
    pseudo_spk_xvecs = np.concatenate(pseudo_spk_xvecs, axis=0)
    print("Pseudo xvecvs: ", pseudo_spk_xvecs.shape)
    log.close()

    return pseudo_xvec_map, pseudo_gender_map, pseudo_spk_xvecs

def draw_ellipse(position, covariance, ax=None, drank=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 3):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        if drank:
            ax.text(position[0], position[1], drank, fontsize=8, color='navy')



src_data = basename(src_data_dir)
pool_data = basename(pool_data_dir)

src_spk2gender_file = join(src_data_dir, 'spk2gender')
src_spk2utt_file = join(src_data_dir, 'spk2utt')
pool_spk2gender_file = join(pool_data_dir, 'spk2gender')

src_spk2gender = {}
src_spk2utt = {}
pool_spk2gender = {}
# Read source spk2gender and spk2utt
print("Reading source spk2gender.")
with open(src_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2gender[sp[0]] = sp[1]
print("Reading source spk2utt.")
with open(src_spk2utt_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2utt[sp[0]] = sp[1:]
# Read pool spk2gender
print("Reading pool spk2gender.")
with open(pool_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        pool_spk2gender[sp[0]] = sp[1]

# Read pool xvectors
print("Reading pool xvectors.")
pool_xvec_file = join(xvec_out_dir, 'xvectors_'+pool_data,
                     'spk_xvector.scp')
pool_xvecs = []
pool_spks = []
pool_gender = []
with ReadHelper('scp:'+pool_xvec_file) as reader:
    for key, xvec in reader:
        #print(key, mat.shape)
        pool_spks.append(key)
        pool_xvecs.append(xvec[np.newaxis])
        pool_gender.append(pool_spk2gender[key])

pool_xvecs = np.concatenate(pool_xvecs)
print("Pool = ", pool_xvecs.shape)
#mean_Xvecs = np.mean(Xvecs, axis=0)
#std_Xvecs = np.std(Xvecs, axis=0)
#Xvecs = (Xvecs - mean_Xvecs) / std_Xvecs
pcolor = []
for spkid in pool_spks:
    c = 'lightcoral' if pool_spk2gender[spkid] == 'f' else 'cornflowerblue'
    pcolor.append(c)

# Read source xvectors
print("Reading source xvectors.")
src_xvec_file = join(xvec_out_dir, 'xvectors_'+src_data,
                     'spk_xvector.scp')
src_xvecs = []
src_spks = []
src_gender = []
with ReadHelper('scp:'+src_xvec_file) as reader:
    for key, xvec in reader:
        src_spks.append(key)
        src_xvecs.append(xvec[np.newaxis])
        src_gender.append(src_spk2gender[key])

src_xvecs = np.concatenate(src_xvecs)
print("SRC = ", src_xvecs.shape)

gmm_dir = join(xvec_out_dir, 'xvectors_'+pool_data, 'gmm')

# Estimate or load existing GMM model
gmm, gmm_meta, xmean, xstd = estimate_density(pool_xvecs, pool_gender, N, gmm_dir)

# Normalize both pool and src xvecs
pool_xvecs = znorm(pool_xvecs, xmean, xstd)
src_xvecs = znorm(src_xvecs, xmean, xstd)

# Sample pseudo-speakers from already learnt distribution
gmm_log = join("gmm-512d-transitions_"+src_data+".log")
pseudo_xvecs, pseudo_gender, pseudo_spk_xvecs = predict_pseudo_xvecs(src_spks, src_spk2utt, src_xvecs, src_gender, gmm, gmm_meta, gmm_log)


# Write features as ark,scp
print("Writing pseudo-speaker xvectors to: "+pseudo_xvec_dir)
ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
                    pseudo_xvec_dir, 'pseudo_xvector',
                    pseudo_xvec_dir, 'pseudo_xvector')
with WriteHelper(ark_scp_output) as writer:
    for uttid, xvec in pseudo_xvecs.items():
        # De-normalizing x-vector to fix the distribution so that
        # it matches the input of NSF module.
        xvec = (xvec * xstd) + xmean
        writer(uttid, xvec)

print("Writing pseudo-speaker spk2gender.")
with open(join(pseudo_xvec_dir, 'spk2gender'), 'w') as f:
    spk2gen_arr = [spk+' '+gender for spk, gender in pseudo_gender.items()]
    sorted_spk2gen = sorted(spk2gen_arr)
    f.write('\n'.join(sorted_spk2gen) + '\n')
