import sys
from os.path import join, basename

from ioTools import readwrite
from kaldiio import WriteHelper, ReadHelper
import numpy as np

args = sys.argv
data_dir = args[1]
anoni_pool = args[2]
pseudo_xvector_dir = args[3]
out_dir = args[4]

dataname = basename(data_dir)
yaap_pitch_dir = join(data_dir, 'yaapt_pitch')
xvec_out_dir = join(out_dir, "xvector")
pitch_out_dir = join(out_dir, "f0")

xvector_file = join(pseudo_xvector_dir, 'pseudo_xvector.scp')
utt2pool_file = join(pseudo_xvector_dir, 'utt2pool')
pool_pitch_dir = join(anoni_pool, 'yaapt_pitch')


def percentile_interpolate(src_f0, tgt_f0):
    src_f0_nz = src_f0[src_f0 != 0]
    tgt_f0_nz = tgt_f0[tgt_f0 != 0]

    # Get source percentile
    src_order = src_f0_nz.argsort()
    src_ranks = np.empty_like(src_order)
    len_src = len(src_f0_nz)
    src_ranks[src_order] = np.arange(len_src)
    src_percentiles = (src_ranks / len_src) * 100

    # Get target pitch corresponding to source percentile
    tgt_values = np.percentile(tgt_f0_nz, src_percentiles)

    ip_f0 = np.zeros(src_f0.shape[0])
    tcnt = 0
    for idx, val in enumerate(src_f0.tolist()):
        if val != 0:
            ip_f0[idx] = tgt_values[tcnt]
            tcnt += 1

    return ip_f0

utt2pool = {}
with open(utt2pool_file) as f:
    for line in f.read().splitlines():
        sp = line.split()
        utt2pool[sp[0]] = sp[1:]


# Write pitch features
pitch_file = join(data_dir, 'pitch.scp')
pitch2shape = {}
with ReadHelper('scp:'+pitch_file) as reader:
    for key, mat in reader:
        pitch2shape[key] = mat.shape[0]
        kaldi_f0 = mat[:, 1].squeeze().copy()
        yaapt_f0 = readwrite.read_raw_mat(join(yaap_pitch_dir, key+'.f0'), 1)
        #unvoiced = np.where(yaapt_f0 == 0)[0]
        #kaldi_f0[unvoiced] = 0
        #readwrite.write_raw_mat(kaldi_f0, join(pitch_out_dir, key+'.f0'))
        f0 = np.zeros(kaldi_f0.shape)
        f0[:yaapt_f0.shape[0]] = yaapt_f0

        # Pitch interpolation based on percentile
        pool_pitch = []
        for pool_spk in utt2pool[key]:
            pool_f0 = readwrite.read_raw_mat(join(pool_pitch_dir, pool_spk+'.f0'), 1)
            pool_pitch.append(pool_f0)
        pool_pitch = np.concatenate(pool_pitch)
        interpolated_f0 = percentile_interpolate(f0, pool_pitch)

        readwrite.write_raw_mat(interpolated_f0, join(pitch_out_dir, key+'.f0'))


# Write xvector features
with ReadHelper('scp:'+xvector_file) as reader:
    for key, mat in reader:
        #print key, mat.shape
        plen = pitch2shape[key]
        mat = mat[np.newaxis]
        xvec = np.repeat(mat, plen, axis=0)
        readwrite.write_raw_mat(xvec, join(xvec_out_dir, key+'.xvector'))


