# user gender
wgender=f


# Anonymization configs
pseudo_xvec_rand_level=spk                # spk (all utterances will have same xvector) or utt (each utterance will have randomly selected xvector)
#cross_gender="same"                      # false, same gender xvectors will be selected; true, other gender xvectors
#cross_gender="other"                      # false, same gender xvectors will be selected; true, other gender xvectors; random gender can be selected
cross_gender="random"                      # false, same gender xvectors will be selected; true, other gender xvectors; random gender can be selected
distance="plda"                           # cosine or plda
#proximity="random"                      # nearest or farthest speaker to be selected for anonymization
#proximity="farthest"                      # nearest or farthest speaker to be selected for anonymization
proximity="dense"                      # nearest or farthest speaker to be selected for anonymization

# other
nj=$(nproc)                          # should not be upper than the number of spekaers in the pool 
