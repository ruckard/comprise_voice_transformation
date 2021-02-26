import sys
from os.path import basename, join
import operator

import numpy as np
import random
from kaldiio import WriteHelper, ReadHelper

from sklearn.externals import joblib

args = sys.argv
print(args)

src_data = args[1]
pool_data = args[2]
affinity_scores_dir = args[3]
xvec_out_dir = args[4]
pseudo_xvecs_dir = args[5]
rand_level = args[6]
cross_gender = args[7]
proximity = args[8]
rand_seed = args[9]

cluster_dir = args[10]

src_affinity_dir = args[11]

# Hyperparams for selecting xvecs in case of far/near
REGION = 100
WORLD = 200

# Hyperparams for selecting clusters in case of dense/sparse
TOPN_DS = 10        # How many clusters to select from density rank
CLUSTER_PROP = 0.5  # Proportion of cluster members to generate pseudo-xvec

random.seed(rand_seed)

if cross_gender == "other":
    print("**Opposite gender speakers will be selected.**")
elif cross_gender == "same":
    print("**Same gender speakers will be selected.**")
elif cross_gender == "random":
    print("**Random gender speakers will be selected.**")


print("Randomization level: " + rand_level)
print("Proximity: " + proximity)

# Core logic of anonymization by randomization
def select_random_xvec(all_spk_list, pool_xvectors):
    # number of random xvectors to select out of pool
    region_mask = random.sample(range(len(all_spk_list)), REGION)
    pseudo_spk_list = [x for i, x in enumerate(all_spk_list) if i in
                       region_mask]
    pseudo_spk_matrix = np.zeros((REGION, 512), dtype='float64')
    for i, spk_aff in enumerate(pseudo_spk_list):
        pseudo_spk_matrix[i, :] = pool_xvectors[spk_aff[0]]
    # Take mean of 100 randomly selected xvectors
    pseudo_xvec = np.mean(pseudo_spk_matrix, axis=0)
    return pseudo_xvec


gender_rev = {'m': 'f', 'f': 'm'}
src_spk2gender_file = join(src_data, 'spk2gender')
src_spk2utt_file = join(src_data, 'spk2utt')
pool_spk2gender_file = join(pool_data, 'spk2gender')

src_spk2gender = {}
src_spk2utt = {}
pool_spk2gender = {}
# Read source spk2gender and spk2utt
src_spks = []
print("Reading source spk2gender.")
with open(src_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spks.append(sp[0])
        src_spk2gender[sp[0]] = sp[1]
print("Reading source spk2utt.")
with open(src_spk2utt_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        src_spk2utt[sp[0]] = sp[1:]
# Read pool spk2gender
print("Reading pool spk2gender.")
pool_spks = []  # to keep track of order of speakers
with open(pool_spk2gender_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        pool_spks.append(sp[0])
        pool_spk2gender[sp[0]] = sp[1]

# For plotting before-after anonymization plots
n_srcspk = len(src_spk2gender)
n_poolspk = len(pool_spk2gender)

# Read pool xvectors
print("Reading pool xvectors.")
pool_xvec_file = join(pool_data, 'xvectors',
                     'spk_xvector.scp')
pool_xvectors = {}
c = 0
XVEC_DIM = 512

n_totalspk = n_srcspk + n_poolspk
X_before = np.zeros((n_totalspk, XVEC_DIM), dtype='float')
X_after = np.zeros((n_totalspk, XVEC_DIM), dtype='float')

with ReadHelper('scp:'+pool_xvec_file) as reader:
    for key, xvec in reader:
        pool_xvectors[key] = xvec
       
        # Add pool speakers for viz
        X_before[c, :] = xvec
        X_after[c, :] = xvec

        XVEC_DIM = xvec.shape[0]
        c += 1
print("Read ", c, "pool xvectors")
print(f"XVEC dimension: {XVEC_DIM}")

assert XVEC_DIM == 512, "Oh No! x-vector dimension is not 512."

# Read src xvectors for visualization purposes
print("Reading src xvectors (mostly for viz).")
src_xvec_file = join(xvec_out_dir, 'xvectors_'+basename(src_data),
                     'spk_xvector.scp')
src_xvectors = {}
c = 0
with ReadHelper('scp:'+src_xvec_file) as reader:
    for key, xvec in reader:
        src_xvectors[key] = xvec

        # Add src speakers for viz
        X_before[n_poolspk+c, :] = xvec
        c += 1
print("Read ", c, "src xvectors")


# Read clustering info
if proximity in ["dense", "sparse"]:
    print("Reading clustering info.")
    pool_affinity_matrix = joblib.load(join(cluster_dir, 'pool_affinity_matrix.pkl'))
    cluster_center_idx = joblib.load(join(cluster_dir, 'cc_idx.pkl'))
    pool_cluster_labels = joblib.load(join(cluster_dir, 'labels.pkl'))
    cluster_center_drank = joblib.load(join(cluster_dir, 'drank.pkl'))

    n_clusters_ = len(cluster_center_idx)

    # Separate male and female cluster centroids and keep track whether they are used or not
    cc_gender = [pool_spk2gender[pool_spks[x]] for x in cluster_center_idx]
    print(f"Male clusters: {cc_gender.count('m')}")
    print(f"Female clusters: {cc_gender.count('f')}")


# These two are main output variables
pseudo_xvec_map = {}
pseudo_gender_map = {}
pseudo_spk_map = {}

for spkidx, spk in enumerate(src_spks):
    gender = src_spk2gender[spk]
    # Filter the affinity pool by gender
    affinity_pool = {}
    # If we are doing cross-gender VC, reverse the gender else gender remains same
    if cross_gender == "other":
        gender = gender_rev[gender]
    elif cross_gender == "random":
        gender = random.choice(list(gender_rev.keys()))

    #print("Filtering pool for spk: "+spk)
    pseudo_gender_map[spk] = gender
    affinity_vec = []
    with open(join(affinity_scores_dir, 'affinity_'+spk)) as f:
        for line in f.read().splitlines():
            sp = line.split()
            pool_spk = sp[1]
            af_score = float(sp[2])
            affinity_vec.append(af_score)
            if pool_spk2gender[pool_spk] == gender:
                affinity_pool[pool_spk] = af_score
    affinity_vec = np.array(affinity_vec)[np.newaxis]


    if proximity == "dense" or proximity == "sparse":
        # Randomly select a dense/sparse cluster among
        # TOPN_DS clusters with same or different gender

        # From affinity vector select the cluster idx closest
        # to this speaker to filter it out
        cidx = affinity_vec[0,cluster_center_idx].argmax()
        closest_cluster_center = cluster_center_idx[cidx]
        print(f"Closest cluster centroid for {spk} is {closest_cluster_center}")

        # If dense, reverse the density ranking, otherwise keep the same order
        if proximity == "dense":
            # Reverse the density ranks and traverse
            rank_iter = reversed(cluster_center_drank)
        elif proximity == "sparse":
            rank_iter = cluster_center_drank

        # Select all clusters from ranklist belonging to prescribed gender and exclude the closest cluster
        gender_clusters = [x for x in rank_iter if cc_gender[x] == gender and x != closest_cluster_center]
        # Keep only TOPN_DS clusters for usage
        gender_clusters = gender_clusters[:TOPN_DS]

        # Select a cluster randomly
        cluster_touse = random.choice(gender_clusters)

        # Use cluster_touse for generating pseudospeaker
        cluster_members = pool_cluster_labels == cluster_touse
        print(f"Using cluster {cluster_touse} which has {sum(cluster_members)} members.")
        # Collect x-vectors of these members and randomly select one as target
        shortlisted_xvecs = []
        shortlisted_idx = []
        for memidx, membership in enumerate(cluster_members):
            if membership:
                shortlisted_xvecs.append(pool_xvectors[pool_spks[memidx]][np.newaxis])
                shortlisted_idx.append(memidx)
        shortlisted_xvecs = np.concatenate(shortlisted_xvecs, axis=0)
        n_shortlisted = shortlisted_xvecs.shape[0]
        print(f"Shortlisted {n_shortlisted} x-vectors from cluster {cluster_touse}.")


        # Select a single x-vector if spk-level otherwise select many
        if rand_level == "spk":
            xvec_idx = random.sample(range(n_shortlisted), int(n_shortlisted * CLUSTER_PROP))
            selected_pool_spks = [pool_spks[shortlisted_idx[ps]] for ps in xvec_idx]
            print(f"Selecting indices {xvec_idx} randomly from {n_shortlisted} xvectors.")
            pseudo_xvec = np.mean(shortlisted_xvecs[xvec_idx, :], axis=0)
            X_after[n_poolspk+spkidx, :] = pseudo_xvec
            # Assign it to all utterances of the current speaker
            for uttid in src_spk2utt[spk]:
                pseudo_xvec_map[uttid] = pseudo_xvec
                pseudo_spk_map[uttid] = selected_pool_spks
        elif rand_level == 'utt':
            # For rand_level = utt, random xvector is assigned to all the utterances
            # of a speaker
            for uttid in src_spk2utt[spk]:
                # Compute random vector for every utt
                xvec_idx = random.sample(range(n_shortlisted), int(n_shortlisted * CLUSTER_PROP))
                selected_pool_spks = [pool_spks[shortlisted_idx[ps]] for ps in xvec_idx]
                pseudo_xvec = np.mean(shortlisted_xvecs[xvec_idx, :], axis=0)
                # Assign it to all utterances of the current speaker
                pseudo_xvec_map[uttid] = pseudo_xvec
                pseudo_spk_map[uttid] = selected_pool_spks
        else:
            print("rand_level not supported! Errors will happen!")


    elif proximity == "farthest" or proximity == "nearest":
        # Sort the filtered affinity pool by scores
        if proximity == "farthest":
            sorted_aff = sorted(affinity_pool.items(), key=operator.itemgetter(1))
        elif proximity == "nearest":
            sorted_aff = sorted(affinity_pool.items(), key=operator.itemgetter(1),
                               reverse=True)


        # Select WORLD least affinity speakers and then randomly select REGION out of
        # them
        top_spk = sorted_aff[:WORLD]
        if rand_level == 'spk':
            # For rand_level = spk, one xvector is assigned to all the utterances
            # of a speaker
            pseudo_xvec = select_random_xvec(top_spk, pool_xvectors)
            X_after[n_poolspk+spkidx, :] = pseudo_xvec
            # Assign it to all utterances of the current speaker
            for uttid in src_spk2utt[spk]:
                pseudo_xvec_map[uttid] = pseudo_xvec
        elif rand_level == 'utt':
            # For rand_level = utt, random xvector is assigned to all the utterances
            # of a speaker
            for uttid in src_spk2utt[spk]:
                # Compute random vector for every utt
                pseudo_xvec = select_random_xvec(top_spk, pool_xvectors)
                # Assign it to all utterances of the current speaker
                pseudo_xvec_map[uttid] = pseudo_xvec
        else:
            print("rand_level not supported! Errors will happen!")

    elif proximity == "random":
        # Baseline - select random REGION speakers from everywhere
        # Regardless of constraints
        whole_list = affinity_pool.items()
        if rand_level == "spk":
            pseudo_xvec = select_random_xvec(whole_list, pool_xvectors)
            X_after[n_poolspk+spkidx, :] = pseudo_xvec
            # Assign it to all utterances of the current speaker
            for uttid in src_spk2utt[spk]:
                pseudo_xvec_map[uttid] = pseudo_xvec
        elif rand_level == 'utt':
            # For rand_level = utt, random xvector is assigned to all the utterances
            # of a speaker
            for uttid in src_spk2utt[spk]:
                # Compute random vector for every utt
                pseudo_xvec = select_random_xvec(whole_list, pool_xvectors)
                # Assign it to all utterances of the current speaker
                pseudo_xvec_map[uttid] = pseudo_xvec


# Write features as ark,scp
print("Writing pseudo-speaker xvectors to: "+pseudo_xvecs_dir)
ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
                    pseudo_xvecs_dir, 'pseudo_xvector',
                    pseudo_xvecs_dir, 'pseudo_xvector')
with WriteHelper(ark_scp_output) as writer:
    for uttid, xvec in pseudo_xvec_map.items():
        writer(uttid, xvec)

print("Writing pseudo-speaker spk2gender.")
with open(join(pseudo_xvecs_dir, 'spk2gender'), 'w') as f:
    spk2gen_arr = [spk+' '+gender for spk, gender in pseudo_gender_map.items()]
    sorted_spk2gen = sorted(spk2gen_arr)
    f.write('\n'.join(sorted_spk2gen) + '\n')

print("Writing exact mapping of pool speakers for pitch conversion.")
with open(join(pseudo_xvecs_dir, 'utt2pool'), 'w') as f:
    utt2pool_arr = [utt + ' ' + ' '.join(pool_list) for utt, pool_list in pseudo_spk_map.items()]
    sorted_utt2pool = sorted(utt2pool_arr)
    f.write('\n'.join(sorted_utt2pool) + '\n')

