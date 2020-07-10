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
import dill
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

    parser.add_argument('input_path', type=str, help='Path to librispeech (root). Only the last one will be transformed', nargs='+')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the transformed utterances',
                        default='output')

    parser.add_argument('-T', '--nb_targets', type=int, help='Max nb of target speakers', default=10)

    args = parser.parse_args()
    method = args.method

    input_paths = args.input_path
    output_path = args.output_path
    nb_proc = multiprocessing.cpu_count() - 1

    nb_targets = args.nb_targets

    for p in input_paths:
        if not os.path.isdir(p):
            raise FileNotFoundError(p)

    os.makedirs(output_path, exist_ok=True)

    if method == 'voicemask':
        from voice_transformation.voicemask import builder
    else:
        from voice_transformation.vtln_based_conversion import builder

    paths = load_librispeech(input_paths)
    print("Nb of speakers=", len(paths))

    nb_targets = min(nb_targets, len(paths))
    print("Nb of target speakers=", nb_targets)
    target_speakers = choose_target(paths, nb_targets)

    with multiprocessing.Pool(nb_proc) as pool:
        print("\nLoad target speakers data")
        target_utterances = {spk_id: load_utterances_parallel([os.path.join(subset, path)
                                                               for subset, path, _ in paths[spk_id]], pool)
                             for spk_id in tqdm.tqdm(target_speakers)}

        print("\nPre-build transformer params from target speakers data")
        transformer_params = builder(target_utterances)

        print("\nSave params with pickle")
        path_to_saved_params = os.path.join(output_path, 'params.pickle')
        with open(path_to_saved_params, 'wb') as f:
            dill.dump(transformer_params, f)
        print("Saved in", path_to_saved_params)


def choose_target(speakers, nb_targets, min_utterances=10):
    speakers_id = list(speakers.keys())
    random.shuffle(speakers_id)
    target_speakers = []
    for spk_id in speakers_id:
        if len(speakers[spk_id]) > min_utterances:
            target_speakers.append(spk_id)
        if len(target_speakers) >= nb_targets:
            break
    return target_speakers


if __name__ == '__main__':
    main()
