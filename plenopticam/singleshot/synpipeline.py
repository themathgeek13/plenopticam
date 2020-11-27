#!/usr/bin/env python

__author__ = "Rohan Rao"
__email__ = "rgrao@andrew.cmu.edu"
__license__ = """
    Copyright (c) 2020 Rohan Rao <rgrao@andrew.cmu.edu>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

def generate_digital_zoom(inputimg, inputdepth, zoom_factor):
    x1A, y1A, __ = np.indices(inputimg.shape)
    H, W, C = inputimg.shape

    u0 = [H//2, W//2]

    x1B = zoom_factor*x1A + (1-zoom_factor)*u0[0]
    x1B = x1B.astype(np.int16).transpose(2,0,1)
    y1B = zoom_factor*y1A + (1-zoom_factor)*u0[1]
    y1B = y1B.astype(np.int16).transpose(2,0,1)

    zoomimg = np.zeros_like(inputimg)
    zoomdepth = np.zeros_like(inputimg)
    for i in range(C):
        zoomimg[x1B[i],y1B[i],i] = inputimg[:, :, i]
        zoomdepth[x1B[i],y1B[i],i] = inputdepth[:, :, i]

    return zoomimg, zoomdepth

class SynthesisPipeline(object):
    
    def __init__(self, *args, **kwargs):
        self.D0 = kwargs['D0'] if 'D0' in kwargs else 0.5
        self.t  = kwargs['t'] if 't' in kwargs else 0.2
        self.u0 = kwargs['u0'] if 'u0' in kwargs else None
        self.maskf = 1000

    def generate_synthesized_views(self, inputimg, inputdepth):
        x1A, y1A, __ = np.indices(inputimg.shape)
        H, W, C = inputimg.shape
        if self.u0 is None:
            self.u0 = [H//2, W//2]

        self.D1A = inputdepth
        x1B = self.D1A*(self.D0-self.t)*x1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[0]/(self.D0*(self.D1A-self.t))
        x1B = x1B.astype(np.uint16).transpose(2,0,1)
        y1B = self.D1A*(self.D0-self.t)*y1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[1]/(self.D0*(self.D1A-self.t))
        y1B = y1B.astype(np.uint16).transpose(2,0,1)

        synthimg = np.ones_like(inputimg)*self.maskf
        synthdepth = np.ones_like(inputdepth)*self.maskf
        for i in range(C):
            synthimg[x1B[i],y1B[i],i] = inputimg[:, :, i]
            synthdepth[x1B[i],y1B[i],i] = inputdepth[:, :, i]

        return synthimg, synthdepth

if __name__ == '__main__':
    I = np.load("/home/rohan/PycharmProjects/plenopticam/data/allfocus.npy")
    D = np.load("/home/rohan/PycharmProjects/plenopticam/data/depthmap.npy")+1.0
    H, W, C = I.shape

    # create the digitally zoomed versions I1 and I2
    I1, D1 = generate_digital_zoom(I, D, zoom_factor=0.95)
    I2, D2 = I.copy(), D.copy()

    # create two separate view synthesis pipelines
    kwargs1 = {'D0': 1, 't': 0, 'u0': [H//2, W//2]}
    sp1 = SynthesisPipeline(**kwargs1)
    I1DZ, D1DZ = sp1.generate_synthesized_views(I1, D1)

    kwargs2 = {'D0': 1, 't': 0.1, 'u0': [H//2, W//2]}
    sp2 = SynthesisPipeline(**kwargs2)
    I2DZ, D2DZ = sp2.generate_synthesized_views(I2, D2)

    # image/depth fusion step
    mask = np.zeros_like(I1DZ)
    mask[np.where(I1DZ == sp1.maskf)] = 1
    I_F = mask*I2DZ + (1-mask)*I1DZ
    D_F = mask*D2DZ + (1-mask)*D1DZ