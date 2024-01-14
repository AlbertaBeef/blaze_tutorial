import numpy as np


from blazebase import BlazeLandmark

class BlazePoseLandmark(BlazeLandmark):
    """The hand landmark model from MediaPipe.
    
    """
    def __init__(self):
        super(BlazePoseLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 256

        #self._define_layers()

    #def _define_layers(self):
    
    #def forward(self, x):


