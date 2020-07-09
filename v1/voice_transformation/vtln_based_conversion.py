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


"""Implementation of a VTLN-based voice conversion method.

This statistical voice conversion method is inspired by the method described in [1].

For a given source speaker S and a given target speaker T and a frequency warping function f,
- cluster the frames of each speaker in "artificial phonetic classes"
- for each source class kS, find the most similar target class kT
- for each pair of classes (kS, kT) find the warping factor that minimises the distance between the "warped" kS and kT

This modules provides:

- a `Transformer` class to transform the utterances of a given speaker
- a `builder` function to pre-build the params needed for the initialisation of the Transformer


See `cls:voice_transformation:VoiceTransformer` for an example

[1] Sundermann, D., & Ney, H. (2003, December). VTLN-based voice conversion. In Proceedings of the 3rd IEEE
    International Symposium on Signal Processing and Information Technology (IEEE Cat. No. 03EX795) (pp. 556-559). IEEE.

"""
import pyworld
import random

import numpy as np
import sklearn.cluster

from voice_transformation import VoiceTransformer, Utterance
from voice_transformation.utils import vtln, pitch


def builder(target_utterances, nb_classes=8, nb_proc=None):
    """Pre-build the parameters to initialize a Transformer

    Parameters
    ----------
    target_utterances: dict
        A dict with speakers ids as keys and utterances of these speakers as values
    nb_classes: int
        Number of "artifical phonetic classes" to consider
    nb_proc: int
        Number of jobs to use for the computation

    Returns
    -------
        A tuple with the built parameters that can be passed to a VTLNBasedTransformer init function as `built_params`:
        - a function to build and fit a Vclustering class
        - the centroids of each target speaker
        - the log f0 of each target speaker

    """
    def k_means(utterances):
        voiced_frames = np.concatenate([utt.spectrogram[utt.voiced_frames] for utt in utterances])
        return sklearn.cluster.KMeans(n_clusters=nb_classes, n_jobs=nb_proc).fit(voiced_frames)

    target_centroids = {spk_id: k_means(target_utterances[spk_id]).cluster_centers_
                        for spk_id in target_utterances}

    target_pitches = {spk_id: pitch.get_log_pitch(target_utterances[spk_id])
                      for spk_id in target_utterances}

    return k_means, target_centroids, target_pitches


class Transformer(VoiceTransformer):
    """Transform speech utterances of a given speaker using vtln-based voice conversion

    Parameters
    ----------
    built_params: tuple
        The params built during preparation step with the builder function
    warping_fn: callable
        The warping function to use. See utils.vtln
    warping_factor_range: list of float
        A list of candidate alpha values
    nb_proc: int
        Number of jobs

    """

    def __init__(self, built_params,
                 warping_fn=vtln.warp_power_function, warping_factor_range=np.array(range(-12, 12, 2)) / 100,
                 nb_proc=None):
        super().__init__()

        # pre-built params
        self.get_clusterer_fn, self.target_centroids, self.target_pitches = built_params

        # local parameters
        self.warping_fn = warping_fn
        self.warping_factor_range = warping_factor_range
        self.nb_proc = nb_proc

        # parameters fitted for the current speaker
        self.source_pitch_ = None
        self.clusterer_ = None
        self.centroids_ = None
        self.warping_factors_ = {}

    def fit(self, utterances):
        """Fit the transformer to a source speaker

        Parameters
        ----------
        utterances: list of Utterance
            Utterances of the source speaker to use to fit the transformer to the source ("enrollment")

        """
        self.source_pitch_ = pitch.get_log_pitch(utterances)
        self.clusterer_ = self.get_clusterer_fn(utterances)
        self.centroids_ = self.clusterer_.cluster_centers_
        for target_id, target_centroids in self.target_centroids.items():
            # learn the mapping between each speaker and his target
            mapping = sklearn.metrics.pairwise_distances_argmin(self.centroids_, target_centroids)
            # learn the warping factor to apply for each class
            warping_factors = []
            for i, source_centroid in enumerate(self.centroids_):
                target_centroid = target_centroids[mapping[i]]
                source_center_warped = [self.warping_fn(alpha)(source_centroid) for alpha in self.warping_factor_range]
                best_alpha_index = sklearn.metrics.pairwise_distances_argmin([target_centroid], source_center_warped)[0]
                warping_factors.append(self.warping_factor_range[best_alpha_index])

            self.warping_factors_[target_id] = warping_factors

    def transform(self, utterance, target=None):
        """Apply transformation to an utterance

        Parameters
        ----------
        utterance: voiced_transformation.Utterance
        target: str or None
            If None, the target will be randomly chosen

        Returns
        -------
        voiced_transformation.Utterance

        """
        if target:
            assert target in self.target_centroids
        else:
            target = random.sample(self.target_centroids.keys(), 1)[0]

        new_spectrogram = utterance.spectrogram.copy()
        for i, frame_class in zip(utterance.voiced_frames,
                                  self.clusterer_.predict(utterance.spectrogram[utterance.voiced_frames])):
            frame = utterance.spectrogram[i]
            new_spectrogram[i] = vtln.vtln_on_frame(frame, self.warping_fn(self.warping_factors_[target][frame_class]))

        new_f0 = pitch.log_gaussian_normalized_f0_conversion(utterance.f0, self.source_pitch_, self.target_pitches[target])

        new_data = pyworld.synthesize(new_f0, new_spectrogram, utterance.aperiodicity,
                                      utterance.sample_rate, utterance.frame_length_in_ms)

        return Utterance(new_data, utterance.sample_rate)
