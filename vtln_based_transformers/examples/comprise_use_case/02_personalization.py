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
import glob
import multiprocessing
import os
import dill

from voice_transformation.utils.load import load_utterances_parallel


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('method', type=str, help='which method ? voicemask or vtln',
                        choices=['voicemask', 'vtln'])

    parser.add_argument('input_path', type=str, help='Path to speaker utterances')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the transformed utterances',
                        default='output')

    parser.add_argument('--params', type=str, help='path to prebuilt params')

    args = parser.parse_args()
    method = args.method

    input_paths = args.input_path
    output_path = args.output_path
    nb_proc = multiprocessing.cpu_count() - 1

    with open(args.params, 'rb') as f:
        transformer_params = dill.load(f)

    os.makedirs(output_path, exist_ok=True)

    if method == 'voicemask':
        from voice_transformation.voicemask import Transformer
    else:
        from voice_transformation.vtln_based_conversion import Transformer

    with multiprocessing.Pool(nb_proc) as pool:
        transformer = Transformer(transformer_params)

        path_to_utterances = glob.glob(input_paths + '/*.flac')
        utterances = load_utterances_parallel(path_to_utterances, pool, desc='load data')

        transformer.fit(utterances)
        path_to_personalized_transformer = os.path.join(output_path, 'personalized_transformer.pickle')
        with open(path_to_personalized_transformer, 'wb') as f:
            dill.dump(transformer, f)
        print("Personalized transformer saved in", path_to_personalized_transformer)


if __name__ == '__main__':
    main()
