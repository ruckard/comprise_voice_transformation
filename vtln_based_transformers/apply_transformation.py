#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# This file is a part of the voice transformation tool
# developed as part of the COMPRISE project
# Author(s): Nathalie Vauquier, Brij Mohan Lal Srivastava
# Copyright (C) 2019 Inria
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import multiprocessing
import os
import random

import numpy as np

from voice_transformation.utils.load import load_utterances_parallel
from voice_transformation.utils.dataset import load_librispeech, load_verbmobil

import tqdm


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('method', type=str, help='which method ? voicemask or vtln',
                        choices=['voicemask', 'vtln'])

    parser.add_argument('corpus', type=str, help='which corpus ? librispeech or verbmobil',
                        choices=['librispeech', 'verbmobil'])

    parser.add_argument('input_path', type=str, help='Path to librispeech (root)', nargs='+')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the transformed utterances',
                        default='output')
    parser.add_argument('-s', '--suffix', type=str, help='suffix to add to the subset name')
    parser.add_argument('-N', '--nb_proc', type=int, help='Nb of parallel processes', default=2)
    parser.add_argument('-T', '--nb_targets', type=int, help='Max nb of target speakers', default=10)
    parser.add_argument('--targets_file', type=str, default='')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    method = args.method
    corpus = args.corpus
    input_paths = args.input_path
    output_path = args.output_path
    suffix = args.suffix
    nb_proc = args.nb_proc
    nb_targets = args.nb_targets
    targets_file = args.targets_file
    resume = args.resume

    for p in input_paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)

    if not suffix:
        suffix = '_mod_' + method

    if resume:
        print('Resuming ...')

    if method == 'voicemask':
        from voice_transformation.voicemask import Transformer, builder
    else:
        from voice_transformation.vtln_based_conversion import Transformer, builder

    if corpus == 'librispeech':
        paths = load_librispeech(input_paths)
    else:
        paths = load_verbmobil(input_paths)
    print("Nb of speakers=", len(paths))

    #####
    # 1. Choose and load data for target speaker(s)
    print("\n1. Target speaker(s)")
    nb_targets = min(nb_targets, len(paths))
    print("Nb of target speakers=", nb_targets)

    print("\n- get target speakers")
    if targets_file:
        print("Load targets list from file")
        target_speakers = load_targets(targets_file)
        assert len(target_speakers) == nb_targets, target_speakers
    else:
        print("Choose targets")
        target_speakers = choose_target(paths, nb_targets)

    with multiprocessing.Pool(nb_proc) as pool:
        print("\n- load target speakers data and pre-build transformer params")
        target_utterances = {spk_id: load_utterances_parallel([os.path.join(subset, path)
                                                               for subset, path, _ in paths[spk_id]], pool)
                             for spk_id in tqdm.tqdm(target_speakers)}
        transformer_params = builder(target_utterances)

        #####
        # 2. Convert utterances
        print("\n2. Conversion\n")
        for spk_id in tqdm.tqdm(paths):
            if not resume or not already_processed(output_path, paths[spk_id], suffix):
                # load all utterances of this speaker
                path_to_utterances = paths[spk_id]
                if spk_id in target_speakers:
                    utterances = target_utterances[spk_id]
                else:
                    utterances = load_utterances_parallel(
                        [os.path.join(subset, path) for subset, path, _ in path_to_utterances],
                        pool, desc='Step 1/2: load data')

                # create the transformer
                transformer = Transformer(transformer_params)
                transformer.fit(utterances)

                # pre-define targets : if we transform utterances of dialogs (Verbmobil), we want to keep the same target
                # for the same speaker for each dialog so we predefine them
                dialog_ids = {dialog_id for _, _, dialog_id in path_to_utterances if dialog_id}
                if dialog_ids:
                    mapping_dialog2target = {dialog_id: target_speakers[np.random.randint(0, nb_targets)]
                                             for dialog_id in dialog_ids}

                    def get_target(dialog):
                        return mapping_dialog2target[dialog]
                else:
                    # if not, the transformer will choose a target
                    def get_target(_):
                        return None

                for (input_subset, path, dialog_id), original_utt in tqdm.tqdm(zip(path_to_utterances, utterances),
                                                                               total=len(utterances), desc='Step 2/2: Conversion'):
                    output_subset = input_subset.split('/')[-1] + '_' + suffix
                    os.makedirs(os.path.join(output_path, output_subset, *path.split('/')[:-1]), exist_ok=True)

                    transformed_utt = transformer.transform(original_utt, target=get_target(dialog_id))
                    transformed_utt.save(os.path.join(output_path, output_subset, path))


def choose_target(speakers, nb_targets, min_utterances=10):
    speakers_id = list(speakers.keys())
    random.shuffle(speakers_id)
    target_speakers = []
    for spk_id in speakers_id:
        if len(speakers[spk_id]) > min_utterances:
            target_speakers.append(spk_id)
        if len(target_speakers) >= nb_targets:
            break

    # save the list of targets in a file
    target_file_path = 'targets.txt'
    if os.path.exists(target_file_path):
        i = 1
        while os.path.exists('targets_{}.txt'.format(i)):
            i += 1
        target_file_path = 'targets_{}.txt'.format(i)

    with open(target_file_path, mode='w') as f:
        for t in target_speakers:
            f.write('{}\n'.format(t))
    print("Targets list saved in", target_file_path)
    return target_speakers


def load_targets(target_file_path):
    with open(target_file_path, mode='r') as f:
        target_speakers = [line.strip() for line in f]
        for line in f:
            print(line)
    return target_speakers


def already_processed(output_path, subset_files, suffix):
    for subset, utt, _ in subset_files:
        new_subset = subset.split('/')[-1] + '_' + suffix
        if not os.path.exists(os.path.join(output_path, new_subset, utt)):
            return False
    return True


if __name__ == '__main__':
    main()
