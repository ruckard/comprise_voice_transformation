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

"""Functions to apply VTLN to a speech signal

3 warping functions are available in this module : power, quadratic and bilinear.
They're provided aas parametrized function : that means that each method warp_[xxx]_function takes an
`alpha` parameter and returns a function that can be called on a frame.

2 functions are provided to apply these warping, respectively on a single frame and on a spectrogram

"""

import scipy.interpolate

import numpy as np


# WARPING FUNCTIONS
# Warning : in this implementation, alpha = 0 means no warping.
# alpha = 0 => no warping
def warp_power_function(alpha):
    def f(omega):
        return np.pi * (omega / np.pi) ** (1 - alpha)
    return f


def warp_quadratic_function(alpha):
    """Returns the quadratic warping function, parametrized with alpha

    The returned function can be apply on a frequency or an array of frequencies, normalized between 0 and pi

    Parameters
    ----------
    alpha: float
        warping factor. 0 means no warp.

    Returns
    -------
    callable
    """

    def f(omega):
        return omega + alpha * (omega / np.pi - (omega / np.pi) ** 2)

    return f


def warp_bilinear_function(alpha):
    """Returns the bilinear warping function, parametrized with alpha

    The returned function can be apply on a frequency or an array of frequencies, normalized between 0 and pi

    Parameters
    ----------
    alpha: float
        warping factor. 0 means no warp.

    Returns
    -------
    callable
    """

    def f(omega):
        z = np.exp(omega * 1j)
        z_t = (z - alpha) / (1 - alpha * z)
        return abs(-1j * np.log(z_t))

        # OR :
        # return omega + 2 * np.arctan((alpha * np.sin(omega)) / (1 - alpha * np.cos(omega)))

    return f


def vtln_on_frame(frame, warping_fn):
    """Apply vtln on a single frame

    Parameters
    ----------
    frame
    warping_fn: callable
        Parametrized warping function

    Returns
    -------

    """
    m = len(frame)
    omega = np.array([((i + 1) * np.pi) / m for i in range(m)])  # [ PI/m, 2PI/m, ... PI ]
    f = scipy.interpolate.interp1d(omega, frame, kind='linear', fill_value='extrapolate')
    omega_warped = warping_fn(omega)
    return f(omega_warped)


def vtln_on_spectrogram(spectrogram, warping_fn):
    """Apply VTLN to a spectrogram

    Parameters
    ----------
    spectrogram: np.array
    warping_fn: callable
        Parametrized warping function

    Returns
    -------
    np.array
    """
    warped_spectrogram = np.empty_like(spectrogram)

    for j, frame in enumerate(spectrogram):
        # warp the frequencies for each frame in the spectrogram
        warped_frame = vtln_on_frame(frame, warping_fn)
        warped_spectrogram[j] = warped_frame

    return warped_spectrogram
