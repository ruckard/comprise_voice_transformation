# Example scripts : COMPRISE use case

These scripts present the 3 steps involved in a voice transformation scenario 
inspired by COMPRISE use case.


## Step 1 : Pre-build the parameters 
This is a preparatory step, ran just once, by the « developer », in laboratory.  
The results of this step is meant to be embedded in an app, for instance.  
This is done with the builder function of the module corresponding to a conversion method. 

Check `01_prebuild_params.py` for an example.  
This script takes as input :
- the conversion method to use
- the path to a dataset with utterances of the speakers that will be used as target speakers 

Example :
```
python 01_prebuild_params.py voicemask ./data/target_speakers
```

## Step 2 : personalization
During this step, a Transformer is initialized with the pre-built params and fit with the voice of the user.
This would run on the device, during the installation, for example.
The result of this step is a `Transformer`, personalized for a user, and ready to be used to transform speech

Check `02_personalization.py` for an example.  
This script takes as input :
- the conversion method to use
- the path to the pre-built params
- the path to samples of the user's speech 

```
python 02_personalization.py --params output/params.pickle voicemask ./data/user/personalization/174/50561
```

## Step 3 : transform an utterance
Finally, the personalized transformer is used to transform the voice of the user..  
This would run on the device.  

Check `03_conversion.py` for an example.  
This script takes as input :
- the transformer to use (from step 2)
- the path to the utterance to transform

```
python 03_conversion.py --transformer output/personalized_transformer.pickle ./data/user/speech/174/168635/174-168635-0000.flac
```