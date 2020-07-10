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

"""Load the data from audio files

The functions of this module load the data and get the features of audio files.

"""

import multiprocessing

import soundfile as sf
import tqdm

from voice_transformation import Utterance


def load_utterance(path, frame_length_in_ms=20, voiced_threshold_factor=0.06, lazy=True):
    """Load an utterance from a path

    Parameters
    ----------
    path: str
        Path to the audio file
    frame_length_in_ms: int
        Length of the frames
    voiced_threshold_factor: float
        Factor to apply to the energy mean to get the voiced threshold
    lazy: bool
        If True, the data will be decoded only when needed. If False, the data will be decoded when loaded.

    Returns
    -------
    Utterance

    """
    with open(path, 'rb') as f:
        data, sample_rate = sf.read(f)

    utterance = Utterance(data, sample_rate,
                          frame_length_in_ms=frame_length_in_ms,
                          voiced_threshold_factor=voiced_threshold_factor)

    if not lazy:
        utterance.decompose()

    return utterance


def load_utterances_parallel(path_to_utterances, pool, desc='Load data'):
    """Load utterances using multiprocessing

    Parameters
    ----------
    path_to_utterances: list of str
        List of paths to audio files
    pool: multiprocessing.Pool
        Pool of processes to run in parallel to decode the utterances

    Returns
    -------
    list of Utterance
    """
    def update_progressbar(q, total):
        progress_bar = tqdm.tqdm(total=total, leave=False, desc=desc)
        while q.get():
            progress_bar.update()

    # Create a queue and a process to display a progressbar
    q = multiprocessing.Manager().Queue()
    p = multiprocessing.Process(target=update_progressbar, args=(q, len(path_to_utterances)))
    p.start()

    # analyse the utterances in parallel
    utterances = pool.starmap(_get_utterance_data, [(path, q) for path in path_to_utterances])

    # stop the progress bar process
    q.put(None)
    # p.close()  # python 3.7
    p.join()
    return utterances


def _get_utterance_data(path, q):
    utt = load_utterance(path, lazy=False)
    q.put(1)  # to display a progressbar
    return utt





