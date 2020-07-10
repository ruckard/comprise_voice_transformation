This project provides:
- classes and functions to perform voice conversion in the voice_transformation library 
- a script to apply them on Librispeech or Verbmobil corpus
- an example of workflow to implement the COMPRISE use case

In this version, 2 voice conversion techniques are available in dedicated modules: voicemask (inspired by [1] and [2]) 
and a vtln-based method (inspired by [3])

For each of these methods, the module consists of:
- a `builder` function to compute some params, specific to each method, from target speakers data. 
- a `Transformer` class to transform the utterances of a speaker. This class has to be:
    - initialized with the pre-built params (and eventual additional parameters)
    - fitted with the speaker voice
 
In the COMPRISE use case:
- the builder would be used to pre-build the parameters to be embedded in the app
- the Transformer class is the part that would be ran on the device 
See [this example](./examples/comprise_use_case/)


[1] Qian, J., Du, H., Hou, J., Chen, L., Jung, T., Li, X. Y., ... & Deng, Y. (2017). Voicemask: Anonymize and sanitize
    voice input on mobile devices. arXiv preprint arXiv:1711.11460.  
[2] Qian, J., Du, H., Hou, J., Chen, L., Jung, T., & Li, X. Y. (2018, November). Hidebehind: Enjoy Voice Input with
    Voiceprint Unclonability and Anonymity. In Proceedings of the 16th ACM Conference on Embedded Networked Sensor
    Systems (pp. 82-94). ACM.  
[3] Sundermann, D., & Ney, H. (2003, December). VTLN-based voice conversion. In Proceedings of the 3rd IEEE
    International Symposium on Signal Processing and Information Technology (IEEE Cat. No. 03EX795) (pp. 556-559). IEEE.

## Install

Requirements:

- numpy
- scipy
- soundfile
- pyworld

Using a conda environment :

```
conda create --name speech scipy numpy tqdm scikit-learn
conda activate speech
pip install soundfile pyworld
```

## Script

A script is available at the root of this repository to run the transformations with pre-defined settings on  
Librispeech or Verbmobil.  
With this script, each utterance of each speaker is converted to an arbitrary target speaker among a group of 
randomly chosen speakers

### Quickstart with voicemask on librispeech
```
python apply_transformation.py voicemask librispeech $LIBRISPEECH_ROOT/dev-clean 
```

This will :
- create a `output` dir in the current directory
- create a subdirectory `dev-clean__mod_voicemask` in it
- replicate the structure (chapter and speakers) of the original `$LIBRISPEECH_ROOT/dev-clean` 
in the new `output/dev-clean__mod_voicemask` folder
- process all the utterances of the dev-clean subset of the Librispeech corpus, and save them in this structure

### Positional arguments
The first positional argument is the method to use : `voicemask` or `vtln`.   
The second one is the name of the corpus (`librispeech` or `verbmobil`).  
The following positional arguments are the paths to the subsets to transform (you can specify several of them)

### Optional arguments
- `-T NB_TARGETS` : the maximum number of target speakers to choose. Since the data of the target speakers 
are kept in memory, this number must be chosen to fit in memory  
- `-N nb_proc` option lets you decide how many jobs you want to use for the parallelized portions of the code. 
- `-o output_path` : where to save the transformed utterances. Default is an `output` folder
created in the current directory.  
- `-s suffix` : which suffix to add to the original subset name. Default is `_mod_voicemask` for 
voicemask, `_mod_vtln` for the VTLN-based conversion
- `--resume` flag let you resume a previously interrupted run
- `--targets_file TARGETS_FILE` : path to a previously created target file to use the same mapping (useful with `--resume`)
