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

"""Pitch conversion"""
import numpy as np


def get_log_pitch(utterances):
    """Get mean and std pitch of a speaker in log domain

    Parameters
    ----------
    utterances: list of Utterances

    """
    log_f0 = np.ma.log(np.concatenate([utt.f0[utt.voiced_frames] for utt in utterances]))
    return log_f0.mean(), log_f0.std()


def log_gaussian_normalized_f0_conversion(f0, source, target):
    """Convert a f0 envelop to a new target speaker pitch

    F0 is converted by using logarithm Gaussian normalized transformation

    Parameters
    ----------
    f0: np.array
        f0 envelop
    source: tuple of float (mean_log, std_log)
    target: tuple of float (mean_log, std_log)
        mean, std pitch of the source and target speakers, in log domain.
        See :func:`get_log_pitch` to get these values from utterances of a speaker

    """
    mean_log_src, std_log_src = source
    mean_log_target, std_log_target = target

    return np.exp(((np.ma.log(f0) - mean_log_src) / std_log_src) * std_log_target + mean_log_target)
