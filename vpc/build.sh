#!/bin/bash
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
anoni_pool=
if [ -z ${anoni_pool} ];then
  anoni_pool=train-other-500
fi

nj=
if [ -z ${nj} ];then
  nj=$(nproc)
fi
stage=1

clear_cache=true

data_url_libritts=www.openslr.org/resources/60     # Link to download LibriTTS corpus
corpora=corpora

. utils/parse_options.sh || exit 1;

. path.sh
. cmd.sh

# x-vector extraction
xvec_nnet_dir=exp/models/2_xvect_extr/exp/xvector_nnet_1a
anon_xvec_out_dir=${xvec_nnet_dir}/anon

#=========== end config ===========

if ${clear_cache}; then
  printf "${GREEN}\nClearing cache...${NC}\n"
  find data/ -iname ".done*" -delete
  find exp/models/ -iname ".done*" -delete
fi

# Download LibriTTS data sets for training anonymization system 
if [ $stage -le 5 ]; then
  printf "${GREEN}\nStage 5: Downloading LibriTTS data sets for training anonymization system ...${NC}\n"
  local/download_and_untar.sh $corpora $data_url_libritts $anoni_pool LibriTTS || exit 1;
fi

libritts_corpus=$(realpath $corpora/LibriTTS)       # Directory for LibriTTS corpus 

# Extract xvectors from anonymization pool

if [ $stage -le 6 ]; then
  # Prepare data for pool 
  printf "${GREEN}\nStage 6: Prepare anonymization pool data...${NC}\n"
  local/data_prep_libritts.sh ${libritts_corpus}/${anoni_pool} data/${anoni_pool} || exit 1;
fi
  
if [ $stage -le 7 ]; then
  printf "${GREEN}\nStage 7: Extracting xvectors for anonymization pool...${NC}\n"
  local/featex/01_extract_xvectors.sh --nj $nj data/${anoni_pool} ${xvec_nnet_dir} \
    ${anon_xvec_out_dir} || exit 1;
fi

echo Done
