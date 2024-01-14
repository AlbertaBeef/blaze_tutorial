import cv2
import numpy as np

import xir
import vitis_ai_library

def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad


def denormalize_detections(detections, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    """
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections


class BlazeBase():
    """ Base class for media pipe models. """

    #def _device(self):
    #    """Which device (CPU or GPU) is being used by this model?"""
    #    return self.classifier_8.weight.device
    
    #def load_weights(self, path):
    #    self.load_state_dict(torch.load(path))
    #    self.eval()  

    def load_xmodel(self, model_path):
        self.DEBUG = False
        
        # Create graph runner
        self.g = xir.Graph.deserialize(model_path)
        self.runner = vitis_ai_library.GraphRunner.create_graph_runner(self.g)

        self.input_tensor_buffers  = self.runner.get_inputs()
        self.output_tensor_buffers = self.runner.get_outputs()

        # Get input scaling
        self.input_fixpos = self.input_tensor_buffers[0].get_tensor().get_attr("fix_point")
        self.input_scale = 2**self.input_fixpos
        print('[BlazeBase] input_fixpos=',self.input_fixpos,' input_scale=',self.input_scale)

        # Get input/output tensors dimensions
        self.inputShape = tuple(self.input_tensor_buffers[0].get_tensor().dims)
        self.outputShape1 = tuple(self.output_tensor_buffers[0].get_tensor().dims)
        self.outputShape2 = tuple(self.output_tensor_buffers[1].get_tensor().dims)
        self.batchSize = self.inputShape[0]
    
        print('[BlazeBase] batch size = ',self.batchSize)
        print('[BlazeBase] input dimensions = ',self.inputShape)
        print('[BlazeBase] output dimensions = ',self.outputShape1,self.outputShape2)


class BlazeLandmark(BlazeBase):
    """ Base class for landmark models. """

    def extract_roi(self, frame, xc, yc, theta, scale):
    #resolution = 256
    #def extract_roi(frame, xc, yc, theta, scale):
        #print("[extract_roi] frame.shape = ", frame.shape," frame.dtype=", frame.dtype)
        #print("[extract_roi] xc.shape=", xc.shape," xc.dtype=", xc.dtype," xc=",xc)
        #print("[extract_roi] yc.shape=", yc.shape," yc.dtype=", yc.dtype," yc=",yc)
        #print("[extract_roi] theta.shape=", theta.shape," theta.dtype=", theta.dtype," theta=",theta)
        #print("[extract_roi] scale.shape=", scale.shape," scale.dtype=", scale.dtype," scale=",scale)

        # Assuming scale is a NumPy array of size [N]
        scaleN = scale.reshape(-1, 1, 1).astype(np.float32)
        #print("[extract_roi] scaleN.shape=", scaleN.shape," scaleN.dtype=", scaleN.dtype)
        #print("[extract_roi] scaleN=", scaleN)

        # Define points
        points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)
        #print("[extract_roi] points.shape=", points.shape," point.dtype=",points.dtype)
        #print("[extract_roi] points=", points)

        # Element-wise multiplication
        points = points * scaleN / 2
        points = points.astype(np.float32)
        #print("[extract_roi] points.shape=", points.shape," point.dtype=",points.dtype)
        #print("[extract_roi] points=", points)

        #R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        R = np.zeros((theta.shape[0],2,2),dtype=np.float32)
        for i in range (theta.shape[0]):
            R[i,:,:] = [[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]]
        #print("[extract_roi] R.shape=", R.shape," R=",R)

        center = np.column_stack((xc, yc))
        #print("[extract_roi] center.shape=", center.shape," center=",center)
        center = np.expand_dims(center, axis=-1)
        #print("[extract_roi] center.shape=", center.shape," center=",center)

        #points = np.matmul(R, points) + center[:, None]
        points = np.matmul(R, points) + center
        points = points.astype(np.float32)
        #print("[extract_roi] points.shape=", points.shape," point.dtype=",points.dtype)
        #print("[extract_roi] points=", points)

        res = self.resolution
        points1 = np.array([[0, 0], [0, res-1], [res-1, 0]], dtype=np.float32)
        #print("[extract_roi] points1.shape=", points1.shape)
        #print("[extract_roi] points1=", points1)

        affines = []
        imgs = []

        for i in range(points.shape[0]):
            pts = points[i,:,:3].T
            #print("[extract_roi] points.shape=", points.shape," points=", points)
            #print("[extract_roi] pts.shape=", pts.shape," pts=", pts)
            #print("[extract_roi] points1.shape=", points1.shape," points1=", points1)
            #print("[extract_roi] pts.dtype=",pts.dtype)
            #print("[extract_roi] points1.dtype=",points1.dtype)
            M = cv2.getAffineTransform(pts, points1)
            img = cv2.warpAffine(frame, M, (res, res))  # No borderValue in NumPy
            img = img.astype('float32') / 255.0
            imgs.append(img)
            affine = cv2.invertAffineTransform(M).astype('float32')
            affines.append(affine)

        if imgs:
            imgs = np.stack(imgs).transpose(0, 3, 1, 2).astype('float32')
            affines = np.stack(affines).astype('float32')
        else:
            imgs = np.zeros((0, 3, res, res), dtype='float32')
            affines = np.zeros((0, 2, 3), dtype='float32')

        return imgs, affines, points


    def denormalize_landmarks(self, landmarks, affines):
        landmarks[:,:,:2] *= self.resolution
        for i in range(len(landmarks)):
            landmark, affine = landmarks[i], affines[i]
            landmark = (affine[:,:2] @ landmark[:,:2].T + affine[:,2:]).T
            landmarks[i,:,:2] = landmark
        return landmarks



class BlazeDetector(BlazeBase):
    """ Base class for detector models.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/
    """
    def load_anchors(self, path):
        self.anchors = np.load(path); #torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        #assert(self.anchors.ndimension() == 2)
        assert(self.anchors.shape[0] == self.num_anchors)
        assert(self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        #return x.float() / 255.# 127.5 - 1.0
        """Converts the image pixels to defined input scale."""
        x = (x / 255.0) * self.input_scale
        x = x.astype(np.int8)

        # Reformat from x of size (256,256,3) to model_x of size (1,256,256,3)
        model_x = []
        model_x.append( x )
        model_x = np.array(model_x)
        
        return model_x        

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        #if isinstance(img, np.ndarray):
        #    img = torch.from_numpy(img).permute((2, 0, 1))

        #return self.predict_on_batch(img.unsqueeze(0))[0]
        
        # Convert img.unsqueeze(0) to NumPy equivalent
        img_expanded = np.expand_dims(img, axis=0)
        #print("[predict_on_image] img.shape=",img.shape)
        #print("[predict_on_image] img_expanded.shape=",img_expanded.shape)

        # Call the predict_on_batch function
        detections = self.predict_on_batch(img_expanded)
        #print("[predict_on_image] len(detections)=",len(detections))

        # Extract the first element from the predictions
        #return predictions[0]        
        if len(detections)>0:
            return np.array(detections)[0]
        else:
            return []


    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        #if isinstance(x, np.ndarray):
        #    x = torch.from_numpy(x).permute((0, 3, 1, 2))

        #assert x.shape[1] == 3
        #assert x.shape[2] == self.y_scale
        #assert x.shape[3] == self.x_scale
        assert x.shape[3] == 3
        assert x.shape[1] == self.y_scale
        assert x.shape[2] == self.x_scale

        # 1. Preprocess the images into tensors:
        #x = x.to(self._device())
        x = self._preprocess(x)
        if self.DEBUG:
            print("[INFO] PalmDetector - prep input buffer ")
        input_data = np.asarray(self.input_tensor_buffers[0])
        input_data[0] = x
                               
        # 2. Run the neural network:
        """ Execute model on DPU """
        if self.DEBUG:
            print("[INFO] PalmDetector - execute ")
        job_id = self.runner.execute_async(self.input_tensor_buffers, self.output_tensor_buffers)
        self.runner.wait(job_id)
                
        output_size=[1,1]
        for i in range(2):
            output_size[i] = int(self.output_tensor_buffers[i].get_tensor().get_element_num() / self.batchSize)
        #print("[INFO] output size :  ",  oself.utput_size )                

        if self.DEBUG:
            print("[INFO] PalmDetector - prep output buffer ")
        out1 = np.asarray(self.output_tensor_buffers[0]) #.reshape(-1,1)
        out2 = np.asarray(self.output_tensor_buffers[1]) #.reshape(-1,18)
                
        # 3. Postprocess the raw predictions:
        if self.DEBUG:
            print("[INFO] PalmDetector - post-processing - extracting detections ")
        detections = self._tensors_to_detections(out2, out1, self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        if self.DEBUG:
            print("[INFO] PalmDetector - post-processing - non-maxima suppression ")
        filtered_detections = []
        for i in range(len(detections)):
            wnms_detections = self._weighted_non_max_suppression(detections[i])
            if len(wnms_detections) > 0:
                filtered_detections.append(wnms_detections)

                if self.DEBUG:
                    print("[INFO] PalmDetector - post-processing - filtered_detections = ",len(filtered_detections),filtered_detections)
                
                if len(filtered_detections) > 0:
                    normalized_detections = np.array(filtered_detections)[0]
                    if self.DEBUG:
                        print("[INFO] PalmDetector - post-processing - normalized_detections = ",normalized_detections.shape,normalized_detections)

        return filtered_detections

    def detection2roi(self, detection):
    #detection2roi_method = 'box'
    #kp1 = 0
    #kp2 = 2
    #theta0 = np.pi/2
    #dscale = 2.6
    #dy = -0.5
    #def detection2roi(detection):
        """ Convert detections from detector to an oriented bounding box.

        Adapted from:
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

        The center and size of the box is calculated from the center 
        of the detected box. Rotation is calcualted from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        """
        if self.detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:,1] + detection[:,3]) / 2
            yc = (detection[:,0] + detection[:,2]) / 2
            scale = (detection[:,3] - detection[:,1]) # assumes square boxes

        elif self.detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            xc = detection[:,4+2*self.kp1]
            yc = detection[:,4+2*self.kp1+1]
            x1 = detection[:,4+2*self.kp2]
            y1 = detection[:,4+2*self.kp2+1]
            scale = ((xc-x1)**2 + (yc-y1)**2).sqrt() * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported"%self.detection2roi_method)

        yc += self.dy * scale
        scale *= self.dscale

        # compute box rotation
        x0 = detection[:,4+2*self.kp1]
        y0 = detection[:,4+2*self.kp1+1]
        x1 = detection[:,4+2*self.kp2]
        y1 = detection[:,4+2*self.kp2+1]
        #theta = torch.atan2(y0-y1, x0-x1) - self.theta0
        theta = np.arctan2(y0-y1, x0-x1) - self.theta0
        #print("[detection2roi] xc.shape=", xc.shape)
        #print("[detection2roi] yc.shape=", yc.shape)
        #print("[detection2roi] theta.shape=", theta.shape)
        #print("[detection2roi] scale.shape=", scale.shape)
        return xc, yc, scale, theta



    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
    #score_clipping_thresh = 100.0
    ##min_score_thresh = 0.5 # too many false positives
    ##min_score_thresh = 0.8 # still has false positives
    #min_score_thresh = 0.7
    #def _tensors_to_detections( raw_box_tensor, raw_score_tensor, anchors):
        if self.DEBUG:
            print("[DEBUG] _tensors_to_detections ... entering")

        detection_boxes = self._decode_boxes(raw_box_tensor, self.anchors)
        #detection_boxes = detection_boxes[0]
            
        if self.DEBUG:
            print("[DEBUG] _tensors_to_detections ... thresholding")

        thresh = self.score_clipping_thresh
        #raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        raw_score_tensor = np.clip(raw_score_tensor,-thresh,thresh)
        #detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        detection_scores = 1/(1 + np.exp(-raw_score_tensor))
        detection_scores = np.squeeze(detection_scores, axis=-1)        

        if self.DEBUG:
            print("[DEBUG] _tensors_to_detections ... min score thresholding (",self.min_score_thresh,")")
        
        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        #mask = detection_scores >= 0.7
        mask = detection_scores >= self.min_score_thresh

        #print(raw_box_tensor.shape, raw_score_tensor.shape)
        #(14, 2944, 18) (14, 2944, 1)
        #print(detection_boxes.shape, detection_scores.shape, mask.shape)
        #(14, 2944, 18) (14, 2944) (14, 2944)         


        if self.DEBUG:
            print("[DEBUG] _tensors_to_detections ... processing loop")

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            #scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            scores = detection_scores[i, mask[i]]
            scores = np.expand_dims(scores,axis=-1)         
            #print(raw_box_tensor.shape,boxes.shape, raw_score_tensor.shape,scores.shape)         
            #(14, 2944, 18) (0, 18) (14, 2944, 1) (0, 1)       
            #output_detections.append(torch.cat((boxes, scores), dim=-1))
            boxes_scores = np.concatenate((boxes,scores),axis=-1)
            #print(boxes_scores.shape)         
            #(0, 19)       
            output_detections.append(boxes_scores)

        if self.DEBUG:
           print("[DEBUG] _tensors_to_detections ... done")

        return output_detections


    def _decode_boxes(self, raw_boxes, anchors):
    #x_scale = 256
    #y_scale = 256
    #h_scale = 256
    #w_scale = 256
    #num_keypoints = 7
    #def _decode_boxes(raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        if self.DEBUG:
            print("[DEBUG] _decode_boxes ... entering")
         
        #boxes = torch.zeros_like(raw_boxes)
        boxes = np.zeros( raw_boxes.shape )

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(self.num_keypoints):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        if self.DEBUG:
            print("[DEBUG] _decode_boxes ... done")

        return boxes

    def _weighted_non_max_suppression(self, detections):
    #num_coords = 18
    #min_suppression_threshold = 0.3
    #def _weighted_non_max_suppression(detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        #remaining = torch.argsort(detections[:, self.num_coords], descending=True)
        remaining = np.argsort(detections[:, self.num_coords])[::-1]    

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            #weighted_detection = detection.clone()
            weighted_detection = np.copy(detection)
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = scores.sum()
                #weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted = np.sum(coordinates * scores, axis=0) / total_score
                weighted_detection[:self.num_coords] = weighted
                weighted_detection[self.num_coords] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    



# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py



# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    #A = box_a.size(0)
    #B = box_b.size(0)
    #max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
    #                   box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
    #                   box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    #inter = torch.clamp((max_xy - min_xy), min=0)
    #return inter[:, :, 0] * inter[:, :, 1]

    # This NumPy version follows a similar approach to the PyTorch code, 
    # using broadcasting and element-wise operations to compute the intersection area. 
    # Note that in NumPy, you use np.minimum, np.maximum, and np.clip 
    # instead of torch.min, torch.max, and torch.clamp. 
    # Also, the unsqueeze operation is replaced with np.expand_dims, 
    # and the expand operation is achieved using repeat.
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = np.minimum(
        np.expand_dims(box_a[:, 2:], axis=1).repeat(B, axis=1),
        np.expand_dims(box_b[:, 2:], axis=0).repeat(A, axis=0)
    )

    min_xy = np.maximum(
        np.expand_dims(box_a[:, :2], axis=1).repeat(B, axis=1),
        np.expand_dims(box_b[:, :2], axis=0).repeat(A, axis=0)
    )

    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

    return inter[:, :, 0] * inter[:, :, 1]    

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A n B / A ? B = A n B / (area(A) + area(B) - A n B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    #inter = intersect(box_a, box_b)
    #area_a = ((box_a[:, 2]-box_a[:, 0]) *
    #          (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    #area_b = ((box_b[:, 2]-box_b[:, 0]) *
    #          (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    #union = area_a + area_b - inter
    #return inter / union  # [A,B]

    # In this NumPy version, unsqueeze is replaced with reshape, 
    # and expand_as is replaced with repeat to handle the expansion along the appropriate dimensions. 
    # The general structure and logic of the code remain the same.
    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).reshape(-1, 1).repeat(box_b.shape[0], axis=1)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).reshape(1, -1).repeat(box_a.shape[0], axis=0)

    union = area_a + area_b - inter

    return inter / union    

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    #return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)
    
    # In this NumPy version, unsqueeze is replaced with np.expand_dims, 
    # and squeeze is used to remove the singleton dimension after calculating the Jaccard overlap 
    # using the previously defined jaccard function. The structure of the function remains the same.
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)

