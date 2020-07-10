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

"""Load the paths to the audio file in datasets

The 2 functions of this module load the paths to the audio files in Librispeech and Verbmobil to
be used as input is the script provided with the library

"""


import glob
import os


def load_librispeech(input_paths):
    """From given input paths, extract the list of speakers and their utterances, according to the librispeech structure

    Parameters
    ----------
    input_paths

    Returns
    -------
    dict
        keys = speaker ids
        values = [tuple(subset,rel path to utterance, None)]. The last value in the tuple is None since librispeech is
        not a dialog corpus
    """
    speakers = {}

    for subset_path in input_paths:
        if subset_path[-1] == '/':
            subset_path = subset_path[:-1]

        if os.path.isdir(subset_path):
            for speaker_id in os.listdir(subset_path):
                speaker_path = os.path.join(subset_path, speaker_id)
                speaker_utterances = []

                if os.path.isdir(speaker_path):
                    for chapter in os.listdir(speaker_path):
                        chapter_path = os.path.join(speaker_path, chapter)

                        if os.path.isdir(chapter_path):
                            speaker_utterances.extend([(subset_path, p.replace(subset_path + '/', ''), None)
                                                       for p in glob.glob(chapter_path + '/*.flac')])
                    speakers[speaker_id] = speaker_utterances

    return speakers


def load_verbmobil(input_paths):
    """From given input paths, extract the list of speakers and their utterances, according to the verbmobil structure

    Parameters
    ----------
    input_paths

    Returns
    -------
    dict
        keys = speaker ids
        dvalues = [tuple(subset,rel path to utterance, dialog identifier)]
    """
    subsets_paths = {}
    for subset_path in input_paths:
        if subset_path[-1] == '/':
            subset_path = subset_path[:-1]
        subset_name = os.path.basename(subset_path)
        subsets_paths[subset_name] = subset_path

    # ids are not unique through VM1 and VM2 : we get true ids from this list
    speakers = {}
    with open(os.path.dirname(__file__) + '/verbmobil.csv') as f:
        for line in f:
            speaker_id, subset_name, wav_path = line.strip().split()
            if subset_name in subsets_paths:
                if speaker_id not in speakers:
                    speakers[speaker_id] = []
                dialog = wav_path.split('/')[1]

                speakers[speaker_id].append((subsets_paths[subset_name], wav_path, dialog))

    return speakers
