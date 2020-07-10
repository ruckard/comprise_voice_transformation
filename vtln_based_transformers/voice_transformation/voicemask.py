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


"""Implementation of a voice transformer based on the Voicemask

This voice conversion method is inspired by the method described in [1] and [2].

In VTLN based voice conversion, a speech signal is transformed by applying a frequency warping function to its
spectrogram.
In this voicemask method, the function is a compound of 2 warping functions (quadratic and bilinear) whose
parameters are randomly chosen.

At the end, the F0 envelop is transformed to the one of a target speaker,
using logarithm Gaussian normalized transformation


This modules provides:

- a `Transformer` class to transform the utterances of a given speaker
- a `builder` function to pre-build the params needed for the initialisation of the Transformer (the mean and std
of the pitches of the target speakers)

See `cls:voice_transformation:VoiceTransformer` for an example

[1] Qian, J., Du, H., Hou, J., Chen, L., Jung, T., & Li, X. Y. (2018, November). Hidebehind: Enjoy Voice Input with
    Voiceprint Unclonability and Anonymity. In Proceedings of the 16th ACM Conference on Embedded Networked Sensor
    Systems (pp. 82-94). ACM.

[2] Qian, J., Du, H., Hou, J., Chen, L., Jung, T., Li, X. Y., ... & Deng, Y. (2017). Voicemask: Anonymize and sanitize
    voice input on mobile devices. arXiv preprint arXiv:1711.11460.


"""

import random

import numpy as np
import pyworld
import scipy.integrate

from voice_transformation import VoiceTransformer, Utterance
from voice_transformation.utils import vtln, pitch


def builder(target_utterances):
    """

    Parameters
    ----------
    target_utterances: dict
        A dict with speakers ids as keys and utterances of these speakers as values

    Returns
    -------
        A tuple with the built parameters that can be passed to a VTLNBasedTransformer init function as `built_params`

    """
    target_pitches = {spk_id: pitch.get_log_pitch(target_utterances[spk_id])
                      for spk_id in target_utterances}

    return target_pitches


class Transformer(VoiceTransformer):
    """Transform speech utterances using voicemask/hidebehind technique

    TODO: doc

    Parameters
    ----------
    built_params: dict
        Params built during the preparation step : see `builder` function.
        Concerning VoiceMask, this is a dict with target (log) pitches
    alpha_range: tuple of float
        The range of the alpha values
    beta_range
        The range of the beta values
    distortion_range
        The range of the distortion we want to achieve

    """

    def __init__(self, built_params, alpha_range=(0.08, 0.10), beta_range=(-2, 2), distortion_range=(0.32, 0.40)):
        super().__init__()

        # pre-built params
        self.target_pitches = built_params
        if self.target_pitches:
            self.pitch_conversion = True
        else:
            self.pitch_conversion = False

        # local parameters
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.distortion_range = distortion_range

        # parameters fitted for the current speaker
        self.source_pitch_ = None

    def fit(self, utterances):
        """

        Parameters
        ----------
        utterances: list of Utterance
            Utterances of the source speaker to use to fit the transformer to the source ("enrollment")

        """
        self.source_pitch_ = pitch.get_log_pitch(utterances)

    def transform(self, utterance, target=None):
        """Apply transformation to an utterance

        Parameters
        ----------
        utterance: voiced_transformation.Utterance
        target: str
            The target is not used to modify the spectrogram, but only for pitch modification
            See utils.pitch.pitch_conversion

        Returns
        -------
        voiced_transformation.Utterance

        """
        alpha = self._choose_alpha()
        beta, compound_function = self._choose_beta(alpha)

        new_spectrogram = vtln.vtln_on_spectrogram(utterance.spectrogram, compound_function)

        if self.pitch_conversion:
            if target:
                assert target in self.target_pitches
            else:
                target = random.sample(self.target_pitches.keys(), 1)[0]
            new_f0 = pitch.log_gaussian_normalized_f0_conversion(utterance.f0, self.source_pitch_, self.target_pitches[target])
        else:
            new_f0 = utterance.f0

        new_data = pyworld.synthesize(new_f0, new_spectrogram,
                                      utterance.aperiodicity, utterance.sample_rate, utterance.frame_length_in_ms)

        return Utterance(new_data, utterance.sample_rate)

    def _choose_alpha(self):
        alpha = (self.alpha_range[1] - self.alpha_range[0]) * np.random.random() + self.alpha_range[0]
        return alpha if random.getrandbits(1) else - alpha

    def _choose_beta(self, alpha):
        candidate_betas = np.arange(*self.beta_range, 0.001)
        random.shuffle(candidate_betas)
        for beta in candidate_betas:
            compound_function = self.compound(alpha, beta)
            dist_strength = self.distortion_strength(compound_function)
            if self.distortion_range[0] <= dist_strength <= self.distortion_range[1]:
                break
        return beta, compound_function

    @staticmethod
    def compound(alpha, beta):
        def f(omega):
            return vtln.warp_quadratic_function(beta)(vtln.warp_bilinear_function(alpha)(omega))

        return f

    @staticmethod
    def distortion_strength(f):
        return scipy.integrate.quad(lambda x: abs(f(x) - x), 0, np.pi)[0]
