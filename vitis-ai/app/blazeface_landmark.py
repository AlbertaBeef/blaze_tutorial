import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blazebase import BlazeLandmark, BlazeBlock

class BlazeFaceLandmark(BlazeLandmark):
    """The face landmark model from MediaPipe.
    
    """
    def __init__(self):
        super(BlazeFaceLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 192

         #self._define_layers()

    #def _define_layers(self):
    
    #def forward(self, x):


