import numpy as np


from blazebase import BlazeDetector


class BlazePose(BlazeDetector):
    """The BlazePose pose detection model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv 
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are 
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """
    def __init__(self):
        super(BlazePose, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 12
        self.score_clipping_thresh = 100.0
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 4

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        self.detection2roi_method = 'alignment'
        # mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
        self.kp1 = 2
        self.kp2 = 3
        self.theta0 = 90 * np.pi / 180
        self.dscale = 1.5
        self.dy = 0.

        #self._define_layers()

    #def _define_layers(self):
    
    #def forward(self, x):


