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
"""Library with basic voice conversion methods

This library was used for deliverable D2.1 of the COMPRISE project to provide baseline performances

This library provides:
- 2 voice conversion methods : VTLN-based and Voicemask. See the documentation of the modules for more info
- basic implementations of vtln and pitch transformation
- utility function to load audio files
- utility function to browse Librispeech and Verbmobil dataset

"""


import numpy as np
import pyworld
import soundfile as sf


class Utterance:
    """Class to get data and features of an utterance
    Attributes
    ----------
    data: np.array
        Data read from the audio file
    sample_rate: int
        Sample rate
    frame_length_in_ms: int
        Frame period to use to decompose this utterance
    voiced_threshold_factor: float
        Threshold to apply to the energy to determine if a frame is voiced or not
    f0: np.array
        From pyworld
    timeaxis: np.array
        From pyworld
    spectrogram: np.array
        From pyworld
    aperiodicity: np.array
        From pyworld
    voiced_frames: np.array
        Indices of the frames identified as voiced

    """
    def __init__(self, data, sample_rate, frame_length_in_ms=20, voiced_threshold_factor=0.06):
        self.data = data
        self.sample_rate = sample_rate
        self.frame_length_in_ms = frame_length_in_ms
        self.voiced_threshold_factor = voiced_threshold_factor

        self._f0 = None
        self._timeaxis = None
        self._spectrogram = None
        self._aperiodicity = None
        self._voiced_frames = None

    def save(self, path):
        with open(path, mode='wb') as f:
            sf.write(f, self.data, self.sample_rate)

    @property
    def f0(self):
        if self._f0 is None:
            self._harvest()
        return self._f0

    @property
    def timeaxis(self):
        if self._timeaxis is None:
            self._harvest()
        return self._timeaxis

    @property
    def spectrogram(self):
        if self._spectrogram is None:
            self._spectrogram = pyworld.cheaptrick(self.data, self.f0, self.timeaxis, self.sample_rate)
        return self._spectrogram

    @property
    def aperiodicity(self):
        if self._aperiodicity is None:
            self._aperiodicity = pyworld.d4c(self.data, self.f0, self.timeaxis, self.sample_rate)
        return self._aperiodicity

    @property
    def voiced_frames(self):
        if self._voiced_frames is None:
            self._voiced_frames = self._get_voiced_frames(self.spectrogram, self.voiced_threshold_factor)
        return self._voiced_frames

    def _harvest(self):
        self._f0, self._timeaxis = pyworld.harvest(self.data, self.sample_rate, frame_period=self.frame_length_in_ms)

    def decompose(self):
        self._f0, self._timeaxis = pyworld.harvest(self.data, self.sample_rate, frame_period=self.frame_length_in_ms)
        self._spectrogram = pyworld.cheaptrick(self.data, self.f0, self.timeaxis, self.sample_rate)
        self._aperiodicity = pyworld.d4c(self.data, self.f0, self.timeaxis, self.sample_rate)

    @classmethod
    def _get_voiced_frames(cls, squared_magnitude_spectrogram, threshold_factor):
        energy = np.sum(squared_magnitude_spectrogram, axis=1)
        threshold = np.mean(energy) * threshold_factor
        return np.asarray(energy > threshold).nonzero()[0]


class VoiceTransformer:
    """Base class for the voice transformers class of this library


    Examples
    --------
    >>> from voice_transformation import Utterance
    >>> from voice_transformation.vtln_based_conversion import builder, Transformer
    >>> import numpy as np
    >>> source_utterances = [Utterance(np.random.random(8000), 1600) for _ in range(5)]
    >>> target1_utterances = [Utterance(np.random.random(8000), 1600) for _ in range(5)]
    >>> target2_utterances = [Utterance(np.random.random(8000), 1600) for _ in range(5)]
    >>> prebuilt_params = builder({'T1': target1_utterances, 'T2': target2_utterances})
    >>> transformer = Transformer(prebuilt_params)
    >>> transformer.fit(source_utterances)
    >>> transformed = transformer.transform(source_utterances[0])


    """
    def __init__(self):
        pass

    def fit(self, utterances):
        pass

    def transform(self, utterance, target=None):
        pass
