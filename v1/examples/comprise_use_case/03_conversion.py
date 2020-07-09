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


import os
import pickle

from voice_transformation.utils.load import load_utterance


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str, help='Path to the utterance to convert')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the transformed utterance',
                        default='output')

    parser.add_argument('-t', '--transformer', type=str, help='path to personalized transformer')

    args = parser.parse_args()

    input_paths = args.input_path
    output_path = args.output_path

    # Load personalized transformer
    with open(args.transformer, 'rb') as f:
        transformer = pickle.load(f)

    # Load utterance to transform
    original_utt = load_utterance(input_paths)

    path_to_transformed = os.path.join(output_path,
                                       '{}_transformed.flac'.format(os.path.basename(input_paths).split('.')[0]))
    transformed_utt = transformer.transform(original_utt)
    transformed_utt.save(path_to_transformed)
    print('Saved in', path_to_transformed)


if __name__ == '__main__':
    main()
