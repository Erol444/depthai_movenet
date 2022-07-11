import marshal
import numpy as np
from pathlib import Path
import depthai as dai
import cv2
import blobconverter

SCRIPT_DIR = Path(__file__).resolve().parent
MOVENET_LIGHTNING_MODEL = SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob"
MOVENET_THUNDER_MODEL = SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob"

# Set ROI for initial cropping
INITIAL_CROP = (0.1, 0.1, 0.9, 0.9)

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type)
    def rectangle(self, frame, p1, p2):
        cv2.rectangle(frame, p1, p2, self.bg_color, 6)
        cv2.rectangle(frame, p1, p2, self.color, 1)

# ROI calculations
xmin = int(INITIAL_CROP[0] * 720 + 280)
ymin = int(INITIAL_CROP[1] * 720)
xmax = int(INITIAL_CROP[2] * 720 + 280)
ymax = int(INITIAL_CROP[3] * 720)
xdelta = xmax - xmin
ydelta = ymax - ymin

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

class Body:
    def __init__(self, scores=None, keypoints_norm=None, score_thresh=None):
        """
        Attributes:
        scores : scores of the keypoints
        keypoints_norm : keypoints normalized ([0,1]) coordinates (x,y) in the squared cropped region
        keypoints : keypoints coordinates (x,y) in pixels in the source image
        score_thresh : score threshold used
        crop_region : cropped region on which the current body was inferred
        """
        self.scores = scores 
        self.keypoints_norm = keypoints_norm
        self.keypoints = []
        for k in keypoints_norm:
            self.keypoints.append((xmin + xdelta * k[0], ymin + ydelta * k[1]))
        self.keypoints = np.array(self.keypoints).astype(np.int32)
        self.score_thresh = score_thresh

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

class MovenetDepthai:
    """
    Movenet body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
    - model: Movenet blob file,
                    - "thunder": the default thunder blob file (see variable MOVENET_THUNDER_MODEL),
                    - "lightning": the default lightning blob file (see variable MOVENET_LIGHTNING_MODEL),
                    - a path of a blob file. It is important that the filename contains 
                    the string "thunder" or "lightning" to identify the tyoe of the model.
    - score_thresh : confidence score to determine whether a keypoint prediction is reliable (a float between 0 and 1).
    - crop : boolean which indicates if systematic square cropping to the smaller side of 
                    the image is done or not,
    - smart_crop : boolen which indicates if cropping from previous frame detection is done or not,
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - stats : True or False, when True, display the global FPS when exiting.            
    """
    def __init__(self, input_src="rgb",
                model=None, 
                score_thresh=0.2,
                crop=False,
                smart_crop = True,
                internal_fps=None,
                stats=True):

        self.model = model 
        self.score_thresh = score_thresh
        
        if model == "lightning":
            self.model = str(MOVENET_LIGHTNING_MODEL)
            self.pd_input_length = 192
        elif model == "thunder":
            self.model = str(MOVENET_THUNDER_MODEL)
            self.pd_input_length = 256
        else:
            self.model = model
            if "lightning" in str(model):
                self.pd_input_length = 192
            else: # Thunder
                self.pd_input_length = 256
        print(f"Using blob file : {self.model}")

        print(f"MoveNet imput size : {self.pd_input_length}x{self.pd_input_length}x3")
        
        self.crop = crop
        self.smart_crop = smart_crop
        self.internal_fps = internal_fps
        self.stats = stats
        
        if input_src is None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = "laconic" in input_src # Camera frames are not sent to the host
            if internal_fps is None:
                if "thunder" in str(model):
                    self.internal_fps = 12
                else:
                    self.internal_fps = 26
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

        else:
            print(f"Input source '{input_src}' is not supported in edge mode !")
            print("Valid input sources: 'rgb', 'rgb_laconic'")
            import sys
            sys.exit()

        # Defines the default crop region (pads the full image from both sides to make it a square image) 
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        
        self.device = dai.Device(self.create_pipeline())
        print("Pipeline started")

        # For debugging
        # self.q_manip_out = self.device.getOutputQueue(name="manip_out", maxSize=1, blocking=False)
   
        self.nb_frames = 0
        self.nb_pd_inferences = 0

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera) 
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setInterleaved(False)
        cam.setFps(self.internal_fps)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.setPreviewSize(2160, 2160)
        
        cam_out = pipeline.create(dai.node.XLinkOut)
        cam_out.setStreamName("cam_out")
        cam.video.link(cam_out.input)

        # Initial ROI cropping
        crop_manip = pipeline.create(dai.node.ImageManip)
        crop_manip.initialConfig.setCropRect(INITIAL_CROP)
        crop_manip.inputImage.setQueueSize(3)
        maxSize = int(2160 * INITIAL_CROP[3] - INITIAL_CROP[1]) * int(2160 * INITIAL_CROP[2] - INITIAL_CROP[0])
        crop_manip.setMaxOutputFrameSize(maxSize*3) # Worst case
        cam.preview.link(crop_manip.inputImage)

        downscale_manip = pipeline.create(dai.node.ImageManip)
        downscale_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        downscale_manip.initialConfig.setResize(self.pd_input_length, self.pd_input_length)
        crop_manip.out.link(downscale_manip.inputImage)

        # For debugging
        # manip_out = pipeline.createXLinkOut()
        # manip_out.setStreamName("manip_out")
        # manip.out.link(manip_out.input)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(str(Path(self.model).resolve().absolute()))
        downscale_manip.out.link(pd_nn.input)
        # pd_nn.input.setQueueSize(1)
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)


        # Define processing script
        processing_script = pipeline.create(dai.node.Script)
        with open("script.py", "r") as f:
            processing_script.setScript(f.read())
        pd_nn.out.link(processing_script.inputs['from_pd_nn'])

        # Define link to send result to host 
        processing_out = pipeline.create(dai.node.XLinkOut)
        processing_out.setStreamName("processing_out")
        processing_script.outputs['to_host'].link(processing_out.input)

        # Crop based on the hand
        manip_script = pipeline.create(dai.node.Script)
        manip_script.setProcessor(dai.ProcessorType.LEON_CSS)
        processing_script.outputs['to_host'].link(manip_script.inputs['pd_in']) # Pose detections
        
        crop_manip.out.link(manip_script.inputs['frame']) # Cropped HQ frame
        manip_script.inputs['frame'].setBlocking(False)
        manip_script.inputs['frame'].setQueueSize(1)

        manip_script.setScript("""
        import marshal
        REGION = 192
        while True:
            frame = node.io['frame'].get()
            node.warn(f"{frame.getWidth()}x{frame.getHeight()}")
            pdin = node.io['pd_in'].get()
            result = marshal.loads(pdin.getData())
            keypoints = list(zip(result["xnorm"], result["ynorm"]))
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            coords = left_wrist if left_wrist[1] < right_wrist[1] else right_wrist

            NORM_REG_X = REGION / frame.getWidth()
            NORM_REG_Y = REGION / frame.getHeight()

            xmin = coords[0] - NORM_REG_X
            ymin = coords[1] - NORM_REG_Y
            if xmin < 0: xmin = 0.0
            if ymin < 0: ymin = 0.0
            xmax = xmin + 2 * NORM_REG_X
            ymax = ymin + 2 * NORM_REG_Y
            if 1 < xmax: xmax = 1.0
            if 1 < ymax: ymax = 1.0
            # node.warn(f"{coords[0]}, {coords[1]}")

            cfg = ImageManipConfig()
            node.warn(f"{xmin}, {ymin}, {xmax}, {ymax}")
            cfg.setCropRect(xmin, ymin, xmax, ymax)
            cfg.setResize(384, 384)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(frame)
            # node.warn(keypoints)
        """)

        qr_manip = pipeline.create(dai.node.ImageManip)
        qr_manip.initialConfig.setResize(384, 384)
        qr_manip.setMaxOutputFrameSize(384 * 384 * 3)
        qr_manip.setWaitForConfigInput(True)
        manip_script.outputs['manip_cfg'].link(qr_manip.inputConfig)
        manip_script.outputs['manip_img'].link(qr_manip.inputImage)

        crop_out = pipeline.create(dai.node.XLinkOut)
        crop_out.setStreamName("crop_out")
        qr_manip.out.link(crop_out.input)
        
        qr_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        qr_nn.setConfidenceThreshold(0.3)
        qr_nn.setBlobPath(blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai", shaves=6))
        qr_manip.out.link(qr_nn.input)

        qr_out = pipeline.create(dai.node.XLinkOut)
        qr_out.setStreamName("qr_out")
        qr_nn.out.link(qr_out.input)


        print("Pipeline created.")

        return pipeline

    def pd_postprocess(self, inference):
        result = marshal.loads(inference.getData())
        scores = np.array(result["scores"])
        keypoints_norm = np.array(list(zip(result["xnorm"], result["ynorm"])))
        body = Body(scores, keypoints_norm, self.score_thresh)
        return body

    def next_frame(self):
        in_video = self.q_video.get()
        frame = cv2.resize(in_video.getCvFrame(), (1280, 720)) # 4k -> 720P (divided by 3)
        # Initial ROI visualization
        frame = cv2.rectangle(frame,
            (int(INITIAL_CROP[0] * 720 + 280), int(INITIAL_CROP[1] * 720)),
            (int(INITIAL_CROP[2] * 720 + 280), int(INITIAL_CROP[3] * 720)),
            (0, 127, 255), 2)
        # Get result from device
        inference = self.q_processing_out.get()
        body = self.pd_postprocess(inference)

        # Statistics
        if self.stats:
            self.nb_frames += 1
            self.nb_pd_inferences += 1
        return frame, body


    def draw(self, frame, body):
        LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
                    [10,8],[8,6],[6,5],[5,7],[7,9],
                    [6,12],[12,11],[11,5],
                    [12,14],[14,16],[11,13],[13,15]]

        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.scores[line[0]] > self.score_thresh and body.scores[line[1]] > self.score_thresh]
        cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
        
        for i,x_y in enumerate(body.keypoints):
            if body.scores[i] > self.score_thresh:
                if i % 2 == 1:
                    color = (0,255,0) 
                elif i == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)
    
    def expandDetection(self, det, percent=2):
        percent /= 100
        det.xmin -= percent
        det.ymin -= percent
        det.xmax += percent
        det.ymax += percent
        if det.xmin < 0: det.xmin = 0
        if det.ymin < 0: det.ymin = 0
        if det.xmax > 1: det.xmax = 1
        if det.ymax > 1: det.ymax = 1

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def run(self):
        self.q_video = self.device.getOutputQueue(name="cam_out")
        self.q_processing_out = self.device.getOutputQueue(name="processing_out")
        q_crop_out = self.device.getOutputQueue(name="crop_out")
        q_qr = self.device.getOutputQueue(name="qr_out")

        c = TextHelper()

        crop_frame = None
        detections = []

        while True:
            # Run movenet on next frame
            frame, body = self.next_frame()
            self.draw(frame, body)
            cv2.imshow("frame", frame)

            if q_crop_out.has():
                crop_frame = q_crop_out.get().getCvFrame()

            if q_qr.has():
                detections = q_qr.get().detections
                print('new detections', len(detections))

            if crop_frame is not None:
                for det in detections:
                    self.expandDetection(det)
                    bbox = self.frameNorm(crop_frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                    c.rectangle(crop_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]))
                    c.putText(crop_frame, f"{int(det.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 20))
                    # if DECODE:
                    #     text = decode(frame, bbox, detector)
                    #     c.putText(frame, text, (bbox[0] + 10, bbox[1] + 40))
                cv2.imshow("Hand Cropped", crop_frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
           