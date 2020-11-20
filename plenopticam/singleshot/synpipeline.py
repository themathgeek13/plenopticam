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

class SynthesisPipeline(object):
    
    def __init__(self, *args, **kwargs):
        self.D0 = kwargs['D0'] if 'D0' in kwargs else 0.5
        self.t  = kwargs['t'] if 't' in kwargs else 0.2
        self.D1A = kwargs['D1A'] if 'D1A' in kwargs else 0.9
        self.u0 = kwargs['u0'] if 'u0' in kwargs else None
        self.zoom_factor = kwargs['zoom_factor'] if 'zoom_factor' in kwargs else 0.95

    def generate_synthesized_views(self, inputimg, inputdepth):
        x1A, y1A, __ = np.indices(inputimg.shape)
        H, W, C = inputimg.shape
        if self.u0 is None:
            self.u0 = inputimg[H//2, W//2]

        x1B = self.D1A*(self.D0-self.t)*x1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[0]/(self.D0*(self.D1A-self.t))
        x1B = x1B.astype(np.uint16).transpose(2,0,1)
        y1B = self.D1A*(self.D0-self.t)*y1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[1]/(self.D0*(self.D1A-self.t))
        y1B = y1B.astype(np.uint16).transpose(2,0,1)

        synthimg = np.zeros_like(inputimg)
        synthdepth = np.zeros_like(inputdepth)
        for i in range(C):
            x1Bch = x1B[i].flatten()
            y1Bch = y1B[i].flatten()

            pos = np.array((x1Bch, y1Bch, np.ones(len(x1Bch))*i))
            pos = pos.astype(np.int)
            synthimg[:,:,i] = inputimg[pos[0], pos[1], pos[2]].reshape((H,W))
            synthdepth[:,:,i] = inputdepth[pos[0], pos[1], pos[2]].reshape((H,W))

        return synthimg, synthdepth

    def generate_digital_zoom(self, inputimg, inputdepth):
        x1A, y1A, __ = np.indices(inputimg.shape)
        H, W, C = inputimg.shape

        if self.u0 is None:
            self.u0 = inputimg[H//2, W//2]

        x1B = self.zoom_factor*x1A + (1-self.zoom_factor)*self.u0[0]
        x1B = x1B.astype(np.uint16).transpose(2,0,1)
        y1B = self.zoom_factor*y1A + (1-self.zoom_factor)*self.u0[1]
        y1B = y1B.astype(np.uint16).transpose(2,0,1)

        zoomimg = np.zeros_like(inputimg)
        zoomdepth = np.zeros_like(inputimg)
        for i in range(C):
            x1Bch = x1B[i].flatten()
            y1Bch = y1B[i].flatten()

            pos = np.array((x1Bch, y1Bch, np.ones(len(x1Bch))*i))
            pos = pos.astype(np.int)
            zoomimg[:,:,i] = inputimg[pos[0], pos[1], pos[2]].reshape((H,W))
            zoomdepth[:,:,i] = inputimg[pos[0], pos[1], pos[2]].reshape((H,W))

        return zoomimg, zoomdepth

if __name__ == '__main__':
    I = np.load("/home/rohan/PycharmProjects/plenopticam/data/allfocus.npy")
    D = np.load("/home/rohan/PycharmProjects/plenopticam/data/depthmap.npy")
    sp = SynthesisPipeline()
    I1, D1 = sp.generate_digital_zoom(I, D)
    I2, D2 = I.copy(), D.copy()
    I1DZ, D2DZ = sp.generate_synthesized_views(I1, D1)
    I2DZ, D2DZ = sp.generate_synthesized_views(I2, D2)