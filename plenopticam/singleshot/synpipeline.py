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
        self.I1 = kwargs['I1'] if 'I1' in kwargs else None
        self.D1 = kwargs['D1'] if 'D1' in kwargs else None
        if self.I1 is None or self.D1 is None:
            raise RuntimeError("Cannot create synthesis pipeline without input image/depth!")
        self.u0 = kwargs['u0'] if 'u0' in kwargs else self.I1[self.I1.shape[0]//2, self.I1.shape[1]//2]

    def generate_synthesized_view(self):
        x1A, y1A, __ = np.indices(self.I1.shape)
        H, W, C = self.I1.shape

        x1B = self.D1A*(self.D0-self.t)*x1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[0]/(self.D0*(self.D1A-self.t))
        x1B = x1B.astype(np.uint16).transpose(2,0,1)
        y1B = self.D1A*(self.D0-self.t)*y1A/(self.D0*(self.D1A-self.t)) + self.t*(self.D1A-self.D0)*self.u0[1]/(self.D0*(self.D1A-self.t))
        y1B = y1B.astype(np.uint16).transpose(2,0,1)

        synthimg = np.zeros_like(self.I1)
        synthdepth = np.zeros_like(self.D1)
        for i in range(C):
            x1Bch = x1B[i].flatten()
            y1Bch = y1B[i].flatten()

            pos = np.array((x1Bch, y1Bch, np.ones(len(x1Bch))*i))
            pos = pos.astype(np.int)
            synthimg[:,:,i] = self.I1[pos[0], pos[1], pos[2]].reshape((H,W))
            synthdepth[:,:,i] = self.D1[pos[0], pos[1], pos[2]].reshape((H,W))

        return synthimg, synthdepth
