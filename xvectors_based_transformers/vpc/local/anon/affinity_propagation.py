#!python
import sys
from os.path import join
from itertools import cycle

import numpy as np
import kaldiio
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.externals import joblib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

args = sys.argv

pool_xvec_dir = args[1]
pool_spk2gender = args[2]
plda_scores_dir = args[3]
out_dir = args[4]

spk2gender = {}
# Read gender of speakers
with open(pool_spk2gender) as f:
    for line in f.read().splitlines():
        sp = line.split()
        spk2gender[sp[0]] = sp[1]

# Read PLDA scores for speakers in pool and create affinity matrix
spk_xvec_file = join(pool_xvec_dir, 'spk_xvector.scp')
spks = []
with open(spk_xvec_file) as f:
    for line in f.read().splitlines():
        spks.append(line.split()[0])
X = []
for spk in spks:
    with open(join(plda_scores_dir, 'affinity_'+spk)) as f:
        lines = f.read().splitlines()
        scores = np.array([float(x.split()[2]) for x in lines], dtype='float')
        X.append(scores[np.newaxis])

X = np.concatenate(X, axis=0)
print("affinity matrix shape: ", X.shape)
# Save pool affinity matrix for later uses
joblib.dump(X, join(out_dir, 'pool_affinity_matrix.pkl'))

# Compute Affinity Propagation
af = AffinityPropagation(affinity='precomputed').fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
joblib.dump(cluster_centers_indices, join(out_dir, 'cc_idx.pkl'))
joblib.dump(labels, join(out_dir, 'labels.pkl'))

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='precomputed')
# Before fitting X in t-SNE, make is a valid distance metric
X = -1.0 * X # To convert affinity to distance
X = X - X.min() # To make all entries positive
Y = tsne.fit_transform(X)

pcolor = []
for spkid in spks:
    c = 'lightcoral' if spk2gender[spkid] == 'f' else 'cornflowerblue'
    pcolor.append(c)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
n_members = []
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    #print(f"Members is class {k}: {sum(class_members)}")
    n_members.append(sum(class_members))
    
    cluster_center = Y[cluster_centers_indices[k]]
    cc_col = pcolor[cluster_centers_indices[k]]

    plt.plot(Y[class_members, 0], Y[class_members, 1], col + '.', markersize=3)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=cc_col,
             markeredgecolor='k', markersize=7)
    #for x in Y[class_members]:
    #    plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# Save density rank
density_rank = np.argsort(np.array(n_members))
joblib.dump(density_rank, join(out_dir, 'drank.pkl'))

plt.savefig(join(out_dir, 'tsne.pdf'), dpi=300)

