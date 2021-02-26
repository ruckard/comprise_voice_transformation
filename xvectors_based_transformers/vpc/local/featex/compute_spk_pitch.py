import sys
from ioTools import readwrite
from os.path import join
import numpy as np

args = sys.argv
data_dir = args[1]

yaap_pitch_dir = join(data_dir, 'yaapt_pitch')
spk2utt = join(data_dir, 'spk2utt')

with open(spk2utt) as f:
    for line in f.read().splitlines():
        sp = line.split()
        spk = sp[0]
        spk_pitch = []
        for uid in sp[1:]:
            yaapt_f0 = readwrite.read_raw_mat(join(yaap_pitch_dir, uid+'.f0'), 1)
            spk_pitch.append(yaapt_f0)
        spk_pitch = np.concatenate(spk_pitch)
        readwrite.write_raw_mat(spk_pitch, join(yaap_pitch_dir, spk+'.f0'))





