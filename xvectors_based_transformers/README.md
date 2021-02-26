# COMPRISE Voice Transformation Tool

This project provides a Voice Transformation Tool to apply the anonymization method described in [1], using x-vectors and neural waveform models. This method has been used as the baseline of the Voice Privacy Challenge 2020.


**As this implementation relies on kaldi and this [CURRENNT tool from NII](https://github.com/nii-yamagishilab/project-CURRENNT-public/tree/3b4648f1f4ec45635c217bbf52be74c54aae3b80), it requires a
CUDA-capable graphics card.**   

The transformation can be run locally using scripts or as a REST service. 
The transformation relies on kaldi and other tools: you can either install each of them in your environment (see below) or you can use the provided docker images.   
  

## Overview
From the VPC 2020 repository :  

![](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020/raw/master/baseline/fig/baseline_git.jpg)


> The baseline system uses several independent models:  
> 1. ASR acoustic model to extract BN features (`1_asr_am`) - trained on LibriSpeech-train-clean-100 and LibriSpeech-train-other-500  
> 2. X-vector extractor (`2_xvect_extr`) - trained on VoxCeleb 1 & 2.  
> 3. Speech synthesis (SS) acoustic model (`3_ss_am`) - trained on LibriTTS-train-clean-100.  
> 4. Neural source filter (NSF) model (`4_nsf`) - trained on LibriTTS-train-clean-100.  




Please visit the [challenge website](https://www.voiceprivacychallenge.org/) for more information about the Challenge and this method.


## Quick start: use the RESTful API in Docker to transform an audio file

Prerequisites: docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) should be installed.  

```
docker run -d --gpus all -p 5000:5000 registry.gitlab.inria.fr/comprise/voice_transformation
```
*Depending on your docker installation, you might have to execute this command with sudo privilege*

The service is running on the port 5000 and responds to the endpoint: http://localhost:5000/vpc

- Method: POST
- Input: original audio file, parameters (JSON) 
- Output: transformed audio file

Here is an example of the way to use it in python:

```
import requests
import json

API_URL = 'http://localhost:5000'  # replace localhost with the proper hostname

# Read the content from an audio file
input_file = 'samples/vctk_p225_003.wav'
with open(input_file, mode='rb') as fp:
    content = fp.read()

# Transformation parameters
params = {'wgender': 'f',
          'cross_gender': 'same',
          'distance': 'plda',
          'proximity': 'random',
          'sample-frequency': 48000}

# Call the service
response = requests.post('{}/vpc'.format(API_URL), data=content, params=json.dumps(params))

# Save the result of the transformation in a new file
result_file = 'transformed.wav'
with open(result_file, mode='wb') as fp:
    fp.write(response.content)

```

## Configuration and parameters

```
transform.sh [--anon_pool <anon_pool_dir>|--sample-frequency <nb>|--cross-gender (same|other|random)|--distance (plda|cosine)|--proximity (dense|farthest|random)] --wgender (m|f) <input_file> <output_dir>
```

- `--wgender` (required): gender of the speaker in the audio file to transform
- `--cross-gender`: gender of the target speakers to select from the pool (same as the original speaker, the other one or randomly)
- `--anon-pool`: path to the anonymization pool of speakers to use
- `--distance`: plda or cosine
- `--proximity`: the strategy to choose the pool of target speakers (dense, farthest or random)
- `<input_file>` input path for the wav file to transform (wav format : RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz) 
- `<output_file>` output path: default is results

The transformer expects audio files sampled at a frequency of 16KHz, but other frequencies can be accepted using the `--sample-frequency` param.

## Use the dockerized environment to run the scripts

At a lower level, the implementation is a Kaldi recipe and can be used as is, for the use cases not fulfilled by the RESTful API (for example, building a new anonymization pool, or for batch processing): the entry points are then a set of scripts, one for each functionality. 
Because the execution environment can be tedious to install, we also provide another docker image with only the execution environment (including kaldi) to run these scripts. This is a way to use it:

From the root folder in this repository:
```
docker run --rm --gpus all \
  -v "$(pwd)"/vpc:/opt/vpc \
  registry.gitlab.inria.fr/comprise/voice_transformation/env \
  vpc/transform.sh --wgender f samples/e0003.wav
```
*Note: depending on your docker installation, you may have to run docker with sudo privileges*

## Manual installation

If you don't use the docker container, you will have to install some components (including kaldi) manually.  This installation process expects that kaldi is already installed and accessible in a `kaldi` folder or symlink from the root of this repo.

From the `env` folder:

- Get the submodules : 

```
git submodule update --init --recursive nii
git submodule update --init --recursive nii_scripts
```

- Install some dependencies using `sudo`

```
sudo apt install sox flac
cd kaldi/tools && sudo extras/install_mkl.sh
```

- `./install.sh`


## COMPRISE use case

### Step 1 : Pre-build the parameters 
This is a preparatory step, ran just once.   
The results of this step is the pool of x-vectors, extracted from a subset of LibriSpeech.  

The `vpc/build.sh` script builds an x-vectors pool from a subset of Librispeech.

This script takes as input the name of a librispeech subset. From the `vpc` folder:

```
./vpc/build.sh --anon_pool [librispeech_subset]
```

or, using the dockerized execution environment:

```
docker run --rm --gpus all \
  -v "$(pwd)"/vpc:/opt/vpc \
  registry.gitlab.inria.fr/comprise/voice_transformation/env \
  ./vpc/build.sh --anon-pool [librispeech_subset]
```

You should replace `[librispeech_subset]` with either `dev-other` or `train-other-500`, for example. You can try the former for a quick test but it might not contain enough speakers for our needs. The latter should be used but the build step can take a while.

This will download the given subset of Librispeech and extract the x-vectors of the speakers of this subset. The x-vectors and pitch data will be stored in `io/anon_pool/[librispeech_subset]`. 

### Step 2 : personalization
[TODO: fit / extract user's x-vector]

### Step 3 : transform an utterance
Finally, the personalized transformer is used to transform the voice of the user.

```
./vpc/transform.sh --wgender m ./vpc/samples/e0003.wav output/
```

or, using docker:

```
docker run --gpus all \
  -v "$(pwd)"/vpc:/opt/vpc \
  registry.gitlab.inria.fr/comprise/voice_transformation \
  ./vpc/transform.sh --wgender m ./vpc/samples/e0003.wav output/
```
  
- the "--wgender" argument is the gender of the original audio file
- the next argument "samples/e0003.wav" is the path to the audio file to process
- the last argument (here "output") is where the result will be stored
Chek the section above to get the list of the parameters.

## General information

For more details about the baseline and data, please see [The VoicePrivacy 2020 Challenge Evaluation Plan](https://www.voiceprivacychallenge.org/docs/VoicePrivacy_2020_Eval_Plan_v1_1.pdf)

For the latest updates in the baseline and evaluation scripts, please visit [News and updates page](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020/wiki/News-and-Updates)


## Developers

### Docker image for development

Developers should use the private docker registry : `registry.gitlab.inria.fr/comprise/development/vpc-transformer`  
*(instead of the public one : `registry.gitlab.inria.fr/comprise/voice_transformation`)*  

### Build and push the docker images

#### Private registry:
Build and push the image with the execution environment:

```
docker build -t registry.gitlab.inria.fr/comprise/development/vpc-transformer/env env  
docker push registry.gitlab.inria.fr/comprise/development/vpc-transformer/env
```

Build and push the image with the service:

```
docker build -t registry.gitlab.inria.fr/comprise/development/vpc-transformer .  
docker push registry.gitlab.inria.fr/comprise/development/vpc-transformer
```

#### Public registry:
Build and push the image with the execution environment:

```
docker build -t registry.gitlab.inria.fr/comprise/voice_transformation/env env  
docker push registry.gitlab.inria.fr/comprise/voice_transformation/env
```

Build and push the image with the service:

```
docker build -t registry.gitlab.inria.fr/comprise/voice_transformation .  
docker push registry.gitlab.inria.fr/comprise/voice_transformation
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
