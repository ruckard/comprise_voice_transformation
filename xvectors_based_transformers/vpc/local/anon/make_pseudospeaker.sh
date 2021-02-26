#!/bin/bash
. path.sh
. cmd.sh

rand_level="spk"
cross_gender="same"
distance="cosine"
proximity="farthest"

rand_seed=2020
tsne_dir=

stage=0

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: "
  echo "  $0 [options] <src-data-dir> <pool-data-dir> <xvector-out-dir> <plda-dir>"
  echo "Options"
  echo "   --rand-level=utt     # [utt, spk] Level of randomness while computing the pseudo-xvectors"
  echo "   --rand-seed=<int>     #  Random seed while computing the pseudo-xvectors"
  echo "   --cross-gender=true     # [true, false] Whether to select same or
                                                   other gender while computing the pseudo-xvectors"
  exit 1;
fi

src_data=$1
pool_data=$2
xvec_out_dir=$3
plda_dir=$4

src_dataname=$(basename $src_data)
pool_dataname=$(basename $pool_data)
src_xvec_dir=${xvec_out_dir}/xvectors_${src_dataname}
pool_xvec_dir=${pool_data}/xvectors

affinity_scores_dir=${src_xvec_dir}/spk_pool_scores
src_affinity_dir=${src_xvec_dir}/spk_pairwise_scores
pseudo_xvecs_dir=${src_xvec_dir}/pseudo_xvecs
cluster_dir=${pool_xvec_dir}/cluster

rm -rf ${affinity_scores_dir} ${pseudo_xvecs_dir}
mkdir -p ${affinity_scores_dir} ${pseudo_xvecs_dir} ${cluster_dir} ${src_affinity_dir}

# Iterate over all the source speakers and generate 
# affinity distribution over anonymization pool
src_spk2gender=${src_data}/spk2gender
pool_spk2gender=${pool_data}/spk2gender

if [ $stage -le 0 ]; then
  if [ "$distance" = "cosine" ]; then
    echo "Computing cosine similarity between source to each pool speaker."
    python local/anon/compute_spk_pool_cosine.py ${src_xvec_dir} ${pool_xvec_dir} \
	    ${affinity_scores_dir} || exit 1;
  elif [ "$distance" = "plda" ]; then
    echo "Computing PLDA affinity scores of each source speaker to each pool speaker."
    cut -d\  -f 1 ${src_spk2gender} | while read s; do
      #echo "Speaker: $s"
      local/anon/compute_spk_pool_affinity.sh ${plda_dir} ${src_xvec_dir} ${pool_xvec_dir} \
	   "$s" "${affinity_scores_dir}/affinity_${s}" || exit 1;
    done

    if [ "$proximity" = "dense" ] || [ "$proximity" = "sparse" ]; then
      echo "Computing PLDA affinity scores of each source speaker to each source speaker."
      cut -d\  -f 1 ${src_spk2gender} | while read s; do
      #echo "Speaker: $s"
        local/anon/compute_spk_pool_affinity.sh ${plda_dir} ${src_xvec_dir} ${src_xvec_dir} \
	   "$s" "${src_affinity_dir}/affinity_${s}" || exit 1;
      done

      # Computing pairwise PLDA need not be calculated again
      expo=${cluster_dir}
      if [ ! -f $expo/.done-cluster ]; then
        echo "Computing PLDA affinity scores of each pool speaker to each pool speaker.This is to create the pairwise score matrix for clustering."
        cut -d\  -f 1 ${pool_spk2gender} | while read s; do
        #echo "Speaker: $s"
          local/anon/compute_spk_pool_affinity.sh ${plda_dir} ${pool_xvec_dir} ${pool_xvec_dir} \
	     "$s" "${cluster_dir}/affinity_${s}" || exit 1;
        done
        python local/anon/affinity_propagation.py ${pool_xvec_dir} ${pool_spk2gender} ${cluster_dir} ${cluster_dir}
        touch $expo/.done-cluster
      fi
    fi
  fi
fi

if [ $stage -le 1 ]; then
  if [ ! -z ${tsne_dir} ]; then
    mkdir -p ${tsne_dir}
  fi
# Filter the scores based on gender and then sort them based on affinity. 
# Select the xvectors of 100 farthest speakers and average them to get pseudospeaker.
  if [ "$distance" = "plda" ] || [ "$distance" = "cosine" ]; then
    # Simple ranking based on plda or cosine distance
    python local/anon/gen_pseudo_xvecs.py ${src_data} ${pool_data} ${affinity_scores_dir} \
   	  ${xvec_out_dir} ${pseudo_xvecs_dir} ${rand_level} ${cross_gender} ${proximity} \
    	  ${rand_seed} ${cluster_dir} ${src_affinity_dir} || exit 1;
  elif [ "$distance" = "gmm" ]; then
    if [ "$proximity" != "dense" ] && [ "$proximity" != "sparse" ]; then
      echo "For distance = gmm, valid proximity is dense or sparse!"
      exit 1;
    fi
    python local/anon/gmm_estimation.py ${src_data} ${pool_data} ${xvec_out_dir} \
	    ${pseudo_xvecs_dir} ${rand_level} ${cross_gender} \
	    ${proximity} ${rand_seed} ${tsne_dir} || exit 1;
  fi
fi

