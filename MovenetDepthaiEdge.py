import marshal
import numpy as np
from pathlib import Path
import depthai as dai
import cv2
import time
import math
from cam_control import CamControl
import webbrowser

SCRIPT_DIR = Path(__file__).resolve().parent
MOVENET_LIGHTNING_MODEL = SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob"
MOVENET_THUNDER_MODEL = SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob"

#-------------[USER CONFIG]-----------

# No cropping by default
CROP_ROI = (0.0, 0.0, 1.0, 1.0) # Set ROI for initial cropping

DEBUG = True # Whether to debug the app (show additional frames, bounding boxes)
DECODE = True # Decode QR code
#-------------------------------------


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords, size = 0.8, thickness = 1):
        cv2.putText(frame, text, coords, self.text_type, size, self.bg_color, thickness * 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, size, self.color, thickness, self.line_type)
    def rectangle(self, frame, p1, p2, color = None):
        if color is None:
            cv2.rectangle(frame, p1, p2, self.bg_color, 6)
            cv2.rectangle(frame, p1, p2, self.color, 1)
        else:
            cv2.rectangle(frame, p1, p2, color, 5)

class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __init__(self, bb = None):
        if bb is not None:
            if isinstance(bb, tuple):
                self.xmin = bb[0]
                self.ymin = bb[1]
                self.xmax = bb[2]
                self.ymax = bb[3]
            else: # object
                self.xmin = bb.xmin
                self.ymin = bb.ymin
                self.xmax = bb.xmax
                self.ymax = bb.ymax

            self.xdelta = self.xmax - self.xmin
            self.ydelta = self.ymax - self.ymin

    def __str__(self) -> str:
        return f"{self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}"
    
    def get_tuple(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    def calculate_new_bb(self, bb):
        bb = BoundingBox(bb)
        return BoundingBox((
            self.xmin + self.xdelta * bb.xmin,
            self.ymin + self.ydelta * bb.ymin,
            self.xmin + self.xdelta * bb.xmax,
            self.ymin + self.ydelta * bb.ymax,
        ))

    def normalize_point(self, x: float, y: float, mult = None):
        new_x = self.xmin + self.xdelta * x
        new_y = self.ymin + self.ydelta * y
        if mult is None:
            return (new_x, new_y)
        else:
            return (int(new_x * mult[0]), int(new_y * mult[1]))

    def normalize_to_frame(self, frame):
        return (
            (int(frame.shape[1] * self.xmin), int(frame.shape[0] * self.ymin)),
            (int(frame.shape[1] * self.xmax), int(frame.shape[0] * self.ymax))
            )

    def crop_frame(self, frame):
        topLeft, bottomRight = self.normalize_to_frame(frame)
        return frame[topLeft[1]: bottomRight[1], topLeft[0]: bottomRight[0]]

    def calc_shape_size(self, shape):
        return math.ceil(self.xdelta * shape[0]) * math.ceil(self.ydelta * shape[1]) * 3


# We send ISP frame (5312x6000) to the device, but use 4k video frame (3840x2160) in the pipeline (crop/detection)
isp_to_video = BoundingBox((0.13855, 0.32, 0.86145, 0.68))

DOWNSCALE_48MP = (885, 1000) # Downscale by 5

# Crop 4k video frame to square to get 1:1 aspect ratio
CROP_SQR = (0.21875, 0, 0.78125, 1.0)
crop_square = isp_to_video.calculate_new_bb(CROP_SQR)
crop_square_manip = BoundingBox(CROP_SQR)

# Crop based on the specified CROP ROI
initial_crop = crop_square.calculate_new_bb(CROP_ROI) # Crop based on INITIAL_CROP
initial_crop_manip = crop_square_manip.calculate_new_bb(CROP_ROI)

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
            self.keypoints.append(initial_crop.normalize_point(k[0], k[1], mult=DOWNSCALE_48MP))
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
    def __init__(self, model=None, score_thresh=0.2):
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
        
        self.device = dai.Device(self.create_pipeline())
        print("Pipeline started")


    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera) 
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_48_MP) # 5312x6000 IMX582
        cam.setIspNumFramesPool(2)
        cam.setVideoNumFramesPool(2)
        cam.initialControl.setManualFocus(63)
        cam.initialControl.setManualExposure(10500, 300)
        cam.setInterleaved(False)
        cam.setFps(5)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        cam.initialControl.setSharpness(0)
        cam.initialControl.setLumaDenoise(0)
        cam.initialControl.setChromaDenoise(4)
        
        camControlIn = pipeline.create(dai.node.XLinkIn)
        camControlIn.setStreamName('cam_control')
        camControlIn.setMaxDataSize(1)
        camControlIn.out.link(cam.inputControl)

        cam_out = pipeline.create(dai.node.XLinkOut)
        cam_out.setStreamName("cam_out")
        cam.isp.link(cam_out.input)

        # Initial ROI cropping. 3840x2160 => Square + InitCropping
        crop_manip = pipeline.create(dai.node.ImageManip)
        crop_manip.initialConfig.setCropRect(initial_crop_manip.get_tuple())
        crop_manip.inputImage.setQueueSize(1)
        crop_manip.inputImage.setBlocking(False)
        crop_manip.setMaxOutputFrameSize(initial_crop_manip.calc_shape_size((3840,2160)))
        crop_manip.setNumFramesPool(2)
        crop_manip.setFrameType(dai.ImgFrame.Type.BGR888p)
        cam.video.link(crop_manip.inputImage)

        crop_manip_out = pipeline.create(dai.node.XLinkOut)
        crop_manip_out.setStreamName("crop_manip_out")
        cam.video.link(crop_manip_out.input)

        downscale_manip = pipeline.create(dai.node.ImageManip)
        downscale_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        downscale_manip.initialConfig.setResize(self.pd_input_length, self.pd_input_length)
        crop_manip.out.link(downscale_manip.inputImage)


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
        
        manip_script.setScript("""
        import marshal
        REGION = 0.1 # For QR Code
        while True:
            pdin = node.io['pd_in'].get()
            result = marshal.loads(pdin.getData())
            keypoints = list(zip(result["xnorm"], result["ynorm"]))
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            coords = left_wrist if left_wrist[1] < right_wrist[1] else right_wrist

            xmin = coords[0] - REGION
            ymin = coords[1] - REGION
            if xmin < 0: xmin = 0.0
            if ymin < 0: ymin = 0.0
            xmax = xmin + 2 * REGION
            ymax = ymin + 2 * REGION
            if 1 < xmax: xmax = 1.0
            if 1 < ymax: ymax = 1.0
            # node.warn(f"{coords[0]}, {coords[1]}")

            cfg = ImageManipConfig()
            # node.warn(f"{xmin}, {ymin}, {xmax}, {ymax}")
            cfg.setCropRect(xmin, ymin, xmax, ymax)
            # cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            # node.warn(keypoints)
        """)

        cfg_out = pipeline.createXLinkOut()
        cfg_out.setStreamName("cfg_out")
        manip_script.outputs['manip_cfg'].link(cfg_out.input)

        print("Pipeline created.")

        return pipeline

    def pd_postprocess(self, inference):
        result = marshal.loads(inference.getData())
        scores = np.array(result["scores"])
        keypoints_norm = np.array(list(zip(result["xnorm"], result["ynorm"])))
        body = Body(scores, keypoints_norm, self.score_thresh)
        return body

    def next_frame(self):
        f = self.q_video.get().getCvFrame()
        self.hq = f
        frame = cv2.resize(f, DOWNSCALE_48MP)
        
        # Initial ROI visualization
        if DEBUG:
            topLeft, bottomRight = isp_to_video.normalize_to_frame(frame)
            frame = cv2.rectangle(frame, topLeft, bottomRight, (127, 255, 255), 1)

            topLeft, bottomRight = crop_square.normalize_to_frame(frame)
            frame = cv2.rectangle(frame, topLeft, bottomRight, (255, 127, 255), 1)

            topLeft, bottomRight = initial_crop.normalize_to_frame(frame)
            frame = cv2.rectangle(frame, topLeft, bottomRight, (0, 127, 255), 1)

        # Get result from device
        inference = self.q_processing_out.get()
        body = self.pd_postprocess(inference)

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

    def decode(self, frame):
        try:
            data, vertices_array, binary_qrcode = self.detector.detectAndDecode(frame)
            if data:
                print("Decoded text", data)
                return data
            else:
                print("Decoding failed")
                return ""
        except:
            print("Exception")
            return ""

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self, frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def run(self):
        self.q_video = self.device.getOutputQueue(name="cam_out")
        self.q_processing_out = self.device.getOutputQueue(name="processing_out")
        q_cfg = self.device.getOutputQueue(name="cfg_out")

        q_control = self.device.getInputQueue("cam_control")
        crop_manip_q = self.device.getOutputQueue(name="crop_manip_out")

        c = TextHelper()
        camControl = CamControl(q_control)

        crop_frame = None
        cfg_bb = None # Cfg for QR code detection

        decText = ""
        decTime = time.time()

        if DECODE:
            self.detector = cv2.QRCodeDetector()

        wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            "./models/detect.prototxt", 
            "./models/detect.caffemodel", 
            "./models/sr.prototxt", 
            "./models/sr.caffemodel"
        )

        while True:
            # Run movenet on next frame
            frame, body = self.next_frame()
            self.draw(frame, body)

            if crop_manip_q.has():
                crop4k = crop_manip_q.get().getCvFrame()
                crop4k = cv2.pyrDown(crop4k)
                cv2.imshow('IMX582 video output', cv2.pyrDown(crop4k))

            if q_cfg.has():
                bb = q_cfg.get().getRaw().cropConfig.cropRect
                cfg_bb = initial_crop.calculate_new_bb(bb) # Crop based on INITIAL_CROP
                if DEBUG:
                    topLeft, bottomRight = cfg_bb.normalize_to_frame(frame)

                    # camControl = dai.CameraControl()
                    # camControl.setAutoExposureRegion(topLeft[0], topLeft[1], bottomRight[0], bottomRight[1])
                    # q_control.send(camControl)

                    c.rectangle(frame, topLeft, bottomRight, (255,127,0))
                    crop_frame = cfg_bb.crop_frame(self.hq)
                    # qrWriter.write(crop_frame)
                    cv2.imshow("QR Code crop", crop_frame)
                    # print(topLeft, bottomRight)
                    if DECODE:
                        text, bbox = wechat_detector.detectAndDecode(crop_frame)
                        if text == () and bbox != ():
                            bbox = [bbox[0].astype(int)]
                            c.rectangle(crop_frame, bbox[0][0], bbox[0][2], color=(0, 0, 255))
                        elif text != () and bbox != ():
                            bbox = [bbox[0].astype(int)]
                            decTime = time.time()
                            decText = str(text[0])
                            c.rectangle(crop_frame, bbox[0][0], bbox[0][2])
                            if decText.startswith("http://") or decText.startswith("https://"):
                                webbrowser.open(decText,new=True)
                                break
                        # else:
                            # print("WeChatQRCode wasn't able to detect any QR codes!")
                            # continue

                        if time.time() - decTime < 0.5:
                            c.putText(frame, decText, (20,60), size=3, thickness=3)

            cv2.imshow("IMX582 ISP output", frame)
            key = cv2.waitKey(1)

            camControl.check_key(key)

            if key == 27 or key == ord('q'):
                break
           