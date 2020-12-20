from src.utils import save_image
from src.cameraCalibration import CameraCalibration
from src.undistortion import Undistortion
from src.gradientThresh import *
from src.colorSpaces import *
from src.perspectiveTransform import PerspectiveTransform
from src.laneFinding import LaneFinding
from src.line import Line
import matplotlib.image as mpimg
import glob
import numpy as np


class AdvancedLaneFinding:
    def __init__(self, calibration_dir, data_dir, chessboard, threshold, colorThreshold, perspectiveTransform,
                 laneFinding, videoMode, saveMode):
        # Directory of calibration images
        self.calibration_dir = calibration_dir

        # Test directory
        self.data_dir = data_dir

        # Video Mode
        self.video_mode = videoMode

        # Save Mode
        self.save_mode = saveMode

        # File Name
        self.file_name = ""

        # Checkerboard Size
        self.chessboard_size = chessboard

        # Camera Calibration
        self.calibrate()

        if not self.video_mode:
            # Load image list
            self.img_list = self.imageList()

        if self.video_mode:
            self.frame_cnt = 0

        # Distortion
        self.ud = Undistortion(self.mtx, self.dist)

        # Thresholding parameters
        self.threshold = threshold
        self.colorThreshold = colorThreshold

        # Perspective Transform
        self.perspectiveTransform = perspectiveTransform
        self.pers_transform = PerspectiveTransform(self.perspectiveTransform)

        # Lane Finding
        self.laneFinding = laneFinding
        self.lane_finding = LaneFinding(self.laneFinding)

    def getImageList(self):
        return self.img_list

    def getImageSize(self, img):
        return img.shape[1], img.shape[0]

    def imageList(self):
        images = glob.glob(self.data_dir + "/*")
        img_list = {}
        for idx, file_name in enumerate(images):
            img = mpimg.imread(file_name)
            img_list[file_name] = img
        return img_list

    def calibrate(self):
        # Compute the camera calibration matrix and distortion coefficients
        calibrate_camera = CameraCalibration(self.chessboard_size, self.calibration_dir)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = calibrate_camera.calibrateCamera()

    def setFileName(self, file_name):
        self.file_name = file_name

    def process(self, img):
        # Setting dummy file_name for video frames
        if self.video_mode:
            self.file_name = "frame_" + str(self.frame_cnt) + ".jpg"

        # Apply undistortion
        undistorted_img = self.ud.undistort(self.data_dir, self.file_name, img)

        # Thresholding parameters
        kernel_size = int(self.threshold["kernel_size"])
        grad_thresh = (my_float(self.threshold["grad_thresh"]["low"]), my_float(self.threshold["grad_thresh"]["high"]))
        mag_thresh = (my_float(self.threshold["mag_thresh"]["low"]), my_float(self.threshold["mag_thresh"]["high"]))
        dir_thresh = (my_float(self.threshold["dir_thresh"]["low"]), my_float(self.threshold["dir_thresh"]["high"]))
        color_thresh = (
            my_float(self.colorThreshold["threshold"]["low"]), my_float(self.colorThreshold["threshold"]["high"]))

        # Apply Thresholding (gradient and color selection)
        gradx = abs_sobel_threshold(undistorted_img, orient='x', sobel_kernel=kernel_size, grad_thresh=grad_thresh)
        grady = abs_sobel_threshold(undistorted_img, orient='y', sobel_kernel=kernel_size, grad_thresh=grad_thresh)
        mag_binary = mag_threshold(undistorted_img, sobel_kernel=kernel_size, mag_thresh=mag_thresh)
        dir_binary = dir_threshold(undistorted_img, sobel_kernel=kernel_size, dir_thresh=dir_thresh)
        s_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hls", channel="s_channel")

        # Combine all thresholds
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

        # Perspective Transform
        img_pt = self.pers_transform.warp(combined, self.file_name, self.data_dir)

        if self.save_mode:
            # Save input images
            save_image(img, self.file_name, self.data_dir, "input")
            save_image(undistorted_img, self.file_name, self.data_dir, "undistorted")
            save_image(gradx, self.file_name, self.data_dir, "gradx")
            save_image(grady, self.file_name, self.data_dir, "grady")
            save_image(mag_binary, self.file_name, self.data_dir, "mag_binary")
            save_image(dir_binary, self.file_name, self.data_dir, "dir_binary")
            save_image(s_binary, self.file_name, self.data_dir, "s_binary")
            save_image(combined, self.file_name, self.data_dir, "combined")
            save_image(img_pt, self.file_name, self.data_dir, "perspective")

        """
        # Lane Finding
        res = self.lane_finding.findLines(img_pt)

        # draw the frames
        (out, l, r, lcr, rcr) = res
        laneOverlay = self.lane_finding.draw(out, l, r, pers_transform, file_name, self.data_dir)
        img = cv2.addWeighted(undistorted_img, 1, laneOverlay, 0.3, 0)
        (lcurve, rcurve) = cv2.curvature(lcr, rcr)
        curvature = 0.5 * (lcurve / 1000 + rcurve / 1000)
        cv2.putText(img, "Radius of Curvature:  " + '{:6.2f}km'.format(curvature), (430, 660), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, [0, 0, 255], 2, cv2.LINE_AA)
        (ll, lr, caroff) = cv2.lanepos(l, r)
        cv2.putText(img, "Distance from Center: " + '{:6.2f}cm'.format(caroff * 100), (430, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, [0, 0, 255], 2, cv2.LINE_AA)
        """

        # Increment video frames in case of video processing
        if self.video_mode:
            self.frame_cnt += 1

        return img_pt
