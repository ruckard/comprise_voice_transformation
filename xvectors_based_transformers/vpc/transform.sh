#!/bin/bash

# Transform script adapted from the 
# Script for The First VoicePrivacy Challenge 2020
#
#
# Copyright (C) 2020  <Brij Mohan Lal Srivastava, Natalia Tomashenko, Xin Wang, Jose Patino,...>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


set -e
cd $(dirname $0)

#===== begin config =======

anon_pool=anon_pool/train_other_500
sample_frequency=16000
wgender=
cross_gender=same
distance=plda
proximity=dense
pseudo_xvec_rand_level=spk

expect_args=4
while [[ $1 == \-\-* ]]; do
  case $1 in
    --anon-pool) shift; anon_pool=$1; shift ;;
    --sample-frequency) shift; sample_frequency=$1; shift ;;
    --sample_frequency) shift; sample_frequency=$1; shift ;;
    --wgender) shift; wgender=$1; expect_args=2; shift ;;
    --cross-gender) shift; cross_gender=$1; shift ;;
    --cross_gender) shift; cross_gender=$1; shift ;;
    --distance) shift; distance=$1; shift ;;
    --proximity) shift; proximity=$1; shift ;;
    --*) echo "$0: invalid option '$1'"; exit 1
  esac
done

if [ $# != $expect_args ]; then
    echo "Usage:"
    echo "  transform.sh [--anon_pool <anon_pool_dir>|--sample-frequency <nb>|--cross-gender (same|other|random)|--distance (plda|cosine)|--proximity (dense|farthest|random)] --wgender (m|f) <input_file> <output_dir>"
    echo "Options:"
    echo "  --wgender (m|f)          # gender of the speaker"
    echo "  --anon-pool <anon_pool>             # path to the anonymization pool to use (must have been built with the ./build.sh script"
    echo "  --cross-gender (same|other|random)  # gender of the xvectors to select from the anon_pool: same as the original speaker, the other one or randomly"
    echo "  --distance (plda|cosine)            # "
    echo "  --proximity (dense|farthest|random) # "
    echo "  --sample-frequency <nb>             # sampling frequency of the utterance to transform (default: 16000 "
    exit 1;
fi


# Possible options and default values
debug_level=1
rand_seed=20
nj=$(nproc)

ipath=$1; shift;
opath=$1; shift;

. cmd.sh
. ../env.sh
. path.sh

# MT: maybe simplify the name in a production regime
printf -v results '%(%Y-%m-%d-%H-%M-%S)T' -1
printf -v outname "$(echo $ipath | sed 's|/|_|g')"
results=$opath/${outname}-${results}.wav
# config_suffix=${pseudo_xvec_rand_level}_${cross_gender}_${distance}_${proximity}
# results="${results}-${config_suffix}"


# Chain model for BN PPG extraction
ppg_type=
ppg_model=exp/models/1_asr_am/exp
ppg_dir=${ppg_model}/nnet3_cleaned


# x-vector extraction
xvec_nnet_dir=exp/models/2_xvect_extr/exp/xvector_nnet_1a # change this to pretrained xvector model downloaded from Kaldi website
anon_xvec_out_dir=${xvec_nnet_dir}/anon
plda_dir=${xvec_nnet_dir}



#=========== end config ===========

function create_dir () {

    
    dir=data/$1
    wavpath=$2
    wavgender=$3
    netcdf=$4
    
    mkdir -p $netcdf || exit 1;


    rm -rf ${dir}
    mkdir -p ${dir}
    id=$(basename $wavpath .wav)

    # Create wav.scp and dummy text
    echo "$id $wavpath" > ${dir}/wav.scp
    echo "$id dummy text" > ${dir}/text

    # Create spk2utt and utt2spk
    spk="spk-$id"
    echo "$spk $id" > ${dir}/spk2utt
    echo "$id $spk" > ${dir}/utt2spk

    # Create spk2gender
    echo "$spk $wavgender" > ${dir}/spk2gender
    
}

function create_dir_batch () {

    
    dir=data/$1
    
    wavgender=$3
    netcdf=$4
    
    mkdir -p $netcdf || exit 1;


    rm -rf ${dir}
    mkdir -p ${dir}
    spk=$(basename $2)
        
    for wavpath in $2/*.wav; do
      id=$(basename $wavpath .wav)

      # Create wav.scp and dummy text
      echo "$id $wavpath" >> ${dir}/wav.scp
      echo "$id dummy text" >> ${dir}/text

      # Create spk2utt and utt2spk
      #spk="spk-$id"
      echo "$id $spk" >> ${dir}/utt2spk
      uttlist="$uttlist $id"
    done
    echo "$spk $uttlist" > ${dir}/spk2utt

    # Create spk2gender
    echo "$spk $wavgender" > ${dir}/spk2gender
    
}


function extract_xvectors ()
{

    data_dir=$1
    nnet_dir=$2
    out_dir=$3

    mfccdir=`pwd`/mfcc
    vaddir=`pwd`/mfcc

    mkdir -p ${out_dir}
    dataname=$(basename $data_dir)

    # Note that train_cmd is defined in cmd.sh (from Kaldi)

    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf \
		       --nj $nj --cmd "$train_cmd" ${data_dir} exp/make_mfcc $mfccdir || exit 1

    utils/fix_data_dir.sh ${data_dir} || exit 1
    
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" ${data_dir} exp/make_vad $vaddir || exit 1

    utils/fix_data_dir.sh ${data_dir} || exit 1

    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" --nj $nj \
					  $nnet_dir ${data_dir} $out_dir/xvectors_$dataname || exit 1

    touch $out_dir/xvectors_$dataname/.done
    
}

function create_new_data_anon ()
{
    anon_data_suffix=_anon
    wav_path=${data_netcdf}/${input_wav_dir}/nsf_output_wav
    new_input_wav_dir=data/${input_wav_dir}${anon_data_suffix}
    if [ -d "$new_input_wav_dir" ]; then
	rm -rf ${new_input_wav_dir}
    fi
    utils/copy_data_dir.sh data/${input_wav_dir} ${new_input_wav_dir}
    [ -f ${new_input_wav_dir}/feats.scp ] && rm ${new_input_wav_dir}/feats.scp
    [ -f ${new_input_wav_dir}/vad.scp ] && rm ${new_input_wav_dir}/vad.scp
    # Copy new spk2gender in case cross_gender vc has been done
    cp ${anon_xvec_out_dir}/xvectors_${input_wav_dir}/pseudo_xvecs/spk2gender ${new_input_wav_dir}/
    awk -v p="$wav_path" '{print $1, "sox", p"/"$1".wav", "-t wav -R -b 16 - |"}' data/${input_wav_dir}/wav.scp > ${new_input_wav_dir}/wav.scp
}



#=========== transformation steps ===========


data_netcdf=$(realpath exp/am_nsf_data)   # directory where features for voice anonymization will be stored

if [ -d "${ipath}" ] ; then
  input_wav_dir=batch 
  create_dir_batch ${input_wav_dir} ${ipath} ${wgender} ${data_netcdf}
else
  input_wav_dir=single_wav
  create_dir ${input_wav_dir} ${ipath} ${wgender} ${data_netcdf}
fi

#=========== remove data from previous run =
rm -Rf ./exp/am_nsf_data/${input_wav_dir}
rm -Rf ./exp/models/1_asr_am/exp/nnet3_cleaned/ppg_${input_wav_dir}/

spk2utt=data/$input_wav_dir/spk2utt
num_spk=$(wc -l < $spk2utt)
[ $nj -gt $num_spk ] && nj=$num_spk

# Extract xvectors from data which has to be anonymized
if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.0: Extracting xvectors for ${input_wav_dir}.${NC}\n"
fi
extract_xvectors data/${input_wav_dir} ${xvec_nnet_dir} ${anon_xvec_out_dir} || exit 1;

# Generate pseudo-speakers for source data
if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.1: Generating pseudo-speakers for ${input_wav_dir}.${NC}\n"
fi
local/anon/make_pseudospeaker.sh --rand-level ${pseudo_xvec_rand_level} \
      				 --cross-gender ${cross_gender} \
				 --distance ${distance} \
				 --proximity ${proximity} \
				 --rand-seed ${rand_seed} \
				 data/${input_wav_dir} \
				 ${anon_pool} \
                                 ${anon_xvec_out_dir} \
				 ${plda_dir} || exit 1;

# Extract pitch for source data
if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.2: Pitch extraction for ${input_wav_dir}.${NC}\n"
fi
local/featex/make_pitch.sh --nj $nj --cmd "$train_cmd" --sample-frequency $sample_frequency data/${input_wav_dir} \
			   exp/make_pitch data/${input_wav_dir}/pitch || exit 1;

# Extract PPGs for source data
if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.3: PPG extraction for ${input_wav_dir}.${NC}\n"
fi
local/featex/extract_ppg.sh --nj $nj --stage 0 \
			    ${input_wav_dir} ${ppg_model} \
			    ${ppg_dir}/ppg_${input_wav_dir} || exit 1;

# Create netcdf data for voice conversion
if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.4: Make netcdf data for VC.${NC}\n"
fi
local/anon/make_netcdf.sh --stage 0 data/${input_wav_dir} \
	                  ${anon_pool} \
			  ${ppg_dir}/ppg_${input_wav_dir}/phone_post.scp \
			  ${anon_xvec_out_dir}/xvectors_${input_wav_dir}/pseudo_xvecs \
			  ${data_netcdf}/${input_wav_dir} || exit 1;

if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.5: Extract melspec from acoustic model for ${input_wav_dir}.${NC}\n"
fi
local/vc/am/01_gen.sh ${data_netcdf}/${input_wav_dir} ${ppg_type} || exit 1;

if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.6: Generate waveform from NSF model for ${input_wav_dir}.${NC}\n"
fi
local/vc/nsf/01_gen.sh ${data_netcdf}/${input_wav_dir} || exit 1;

if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.7: Creating new data directories corresponding to anonymization.${NC}\n"
fi
create_new_data_anon 




if [ ! -z ${opath} ];then
  cp ${data_netcdf}/${input_wav_dir}/nsf_output_wav/*.wav ${opath}
else
  cp ${data_netcdf}/${input_wav_dir}/nsf_output_wav/*.wav ${results}
fi

exit 0 
