# COMPRISE Voice Transformation Tool

This project provides a Voice Transformation Tool to apply the anonymization method described in [1], using x-vectors and neural waveform models. This method has been used as the baseline of the Voice Privacy Challenge 2020.


**As this implementation relies on kaldi and this [CURRENNT tool from NII](https://github.com/nii-yamagishilab/project-CURRENNT-public/tree/3b4648f1f4ec45635c217bbf52be74c54aae3b80), it requires a
CUDA-capable graphics card.**   


The transformation can be run locally on a single wav file using script or a REST service.   
A docker image is provided.  

## Overview
From the VPC 2020 repository :  

![](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020/raw/master/baseline/fig/baseline_git.jpg)


> The baseline system uses several independent models:  
> 1. ASR acoustic model to extract BN features (`1_asr_am`) - trained on LibriSpeech-train-clean-100 and LibriSpeech-train-other-500  
> 2. X-vector extractor (`2_xvect_extr`) - trained on VoxCeleb 1 & 2.  
> 3. Speech synthesis (SS) acoustic model (`3_ss_am`) - trained on LibriTTS-train-clean-100.  
> 4. Neural source filter (NSF) model (`4_nsf`) - trained on LibriTTS-train-clean-100.  




Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge and this method.

## Quick Start with Docker

Clone this repo without the submodules ( `git clone [repo-url]` ) so the `io` folder can be mounted as a volume to the docker container that will be used. Models, config and a sample input are placed there, in dedicated subfolders (resp. `exp/models`, `config` and `inputs`).

### Build the x-vectors pool (run once)

A `build.sh` script, inside the container, allows you to build an x-vectors pool from a subset of Librispeech.

From the root of the repo, run :

```
docker run --gpus all \
  -v "$(pwd)"/io:/opt/io \
  registry.gitlab.inria.fr/comprise/development/vpc-transformer \
  ./build.sh --anoni_pool [librispeech_subset]
```

This will download the given subset of Librispeech and extract the x-vectors of the speakers of this dataset. These x-vectors will be used to create a Transformer in the next step.

Note : You should replace `[librispeech_subset]` with either `dev-other` or `train-other-500`. You can try the former for a quick test but it might not contain enough speakers for our needs. The latter should be used but the build step can take a while.


### Running a transformation sample

Edit the `anoni_pool` property in `io/config/config_transform.sh` to use the x-vectors pool you built in the previous step.

```
docker run --gpus all \
  -v "$(pwd)"/io:/opt/io \
  registry.gitlab.inria.fr/comprise/development/vpc-transformer \
  ./transform.sh --ipath io/inputs/e0003.wav
```
*(or replace `e0003.wav` with your own wav file)*
 
The output will be in `io/results`.

The transformation uses the provided pre-trained models and a specific configuration file that sets some transformation parameters (`io/config/config_params.sh`).

You can also run the container in interactive mode :

```
docker run -it --gpus all \
  -v "$(pwd)"/io:/opt/io \
  registry.gitlab.inria.fr/comprise/development/vpc-transformer
```
### Running a REST server to perform the transformation

```
docker run --gpus all \
  -v "$(pwd)"/io:/opt/io \
  -p 5000:5000 \
  registry.gitlab.inria.fr/comprise/development/vpc-transformer \
  python3 app.py
```

Then you can call the transformer at `http://[hostname]:5000/vpc`


## Configuration and parameters

Main parameter for `transform.sh`: 

- `--ipath` input path for the wav file to transform (wav format : RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz) 

- `--opath` output path: default is results

The file `config_transform.sh` contains the parameters of the transformation that can also be given on the command line. 

```
wgender=f                             # gender m or f
pseudo_xvec_rand_level=spk            # spk (all utterances will have same xvector) or utt (each utterance will have randomly selected xvector)
#cross_gender="same"                  # false, same gender xvectors will be selected; true, other gender xvectors
#cross_gender="other"                 # false, same gender xvectors will be selected; true, other gender xvectors; random gender can be selected
cross_gender="random"                 # false, same gender xvectors will be selected; true, other gender xvectors; random gender can be selected
distance="plda"                       # cosine or plda
#proximity="random"                   # nearest or farthest speaker to be selected for anonymization
#proximity="farthest"                 # nearest or farthest speaker to be selected for anonymization
proximity="dense"                     # nearest or farthest speaker to be selected for anonymization
anoni_pool="train_other_500"          # change this to the data you want to use for anonymization pool
#anoni_pool="dev-other"
```


## Manual installation

If you don't use the docker container, you will have to install some components (including kaldi) manually.

### Install

1. Clone this repo with the submodules : `git clone --recurse-submodules https://gitlab.inria.fr/comprise/development/vpc-transformer `
2. You must install some dependencies using `sudo` (or `sudo-g5k` on grid5000). `sudo apt install sox flac`, and `cd kaldi/tools && sudo extras/install_mkl.sh`.
3. ./install.sh


### Running the recipe

Pretrained models can be downloaded with `./baseline/local/download_models.sh` (requires a password from the Voice Privacy Challenge team)

1. `cd vpc` 
2. run `./build.sh` and `transform.sh`. 

## General information

For more details about the baseline and data, please see [The VoicePrivacy 2020 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2020_Eval_Plan_v1_1.pdf)

For the latest updates in the baseline and evaluation scripts, please visit [News and updates page](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020/wiki/News-and-Updates)


## Build the docker image

```
git clone --recurse-submodules https://gitlab.inria.fr/comprise/development/vpc-transformer  
cd vpc-transformer
sudo docker build -t comprise-vpc .
```


## License

Copyright (C) 2020

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

---------------------------------------------------------------------------
