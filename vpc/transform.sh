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

#===== begin config =======

# input file or path
ipath=
opath=

# Possible options and default values
debug_level=1
rand_seed=20
nj=$(nproc)

# load configuration of the transformer (made with init)
. config_transform.sh

# MT not sure it is still useful to parse args if we just read $1 as the input file
. utils/parse_options.sh || exit 1;

. path.sh
. cmd.sh


# MT: maybe simplify the name in a production regime
printf -v results '%(%Y-%m-%d-%H-%M-%S)T' -1
printf -v outname "$(echo $ipath | sed 's|/|_|g')"
results=results/${outname}-${results}.wav
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

#=========== remove data from previous run =
rm -Rf ./exp/am_nsf_data/single_wav
rm -Rf ./exp/models/1_asr_am/exp/nnet3_cleaned/ppg_single_wav/

#=========== transformation steps ===========


data_netcdf=$(realpath exp/am_nsf_data)   # directory where features for voice anonymization will be stored

# MT: extend from a single file to a batch of files?
input_wav_dir=single_wav
create_dir ${input_wav_dir} ${ipath} ${wgender} ${data_netcdf}


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
				 data/${anoni_pool} \
				 ${anon_xvec_out_dir} \
				 ${plda_dir} || exit 1;

# Extract pitch for source data
if [ $debug_level -ge 1 ]; then 
    printf "${RED}\nStage a.2: Pitch extraction for ${input_wav_dir}.${NC}\n"
fi
local/featex/make_pitch.sh --nj $nj --cmd "$train_cmd" data/${input_wav_dir} \
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
			  ${ppg_dir}/ppg_${input_wav_dir}/phone_post.scp \
			  ${anon_xvec_out_dir}/xvectors_${input_wav_dir}/pseudo_xvecs/pseudo_xvector.scp \
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
  cp ${data_netcdf}/${input_wav_dir}/nsf_output_wav/$(basename $ipath) ${opath}
else
  cp ${data_netcdf}/${input_wav_dir}/nsf_output_wav/$(basename $ipath) ${results}
fi

exit 0 
