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
                 laneFinding, videoMode, saveMode, testImagesMode):
        # Directory of calibration images
        self.calibration_dir = calibration_dir

        # Test directory
        self.data_dir = data_dir

        # Video Mode
        self.video_mode = videoMode

        # Save Mode
        self.save_mode = saveMode

        # Test Images Mode
        self.test_images_mode = testImagesMode

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
        self.pers_transform = PerspectiveTransform(self.perspectiveTransform, saveMode)

        # Lane Finding
        self.laneFinding = laneFinding
        self.lane_finding = LaneFinding(self.laneFinding, self.test_images_mode)

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

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

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
        h_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hls", channel="h_channel")
        l_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hls", channel="l_channel")
        s_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hls", channel="s_channel")

        _h_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hsv", channel="h_channel")
        _s_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hsv", channel="s_channel")
        _v_binary = colorspace_select(undistorted_img, thresh=color_thresh, colorspace="hsv", channel="v_channel")

        # Combine all thresholds
        color_binary = np.dstack((np.zeros_like(gradx), gradx, s_binary)) * 255
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
        """
        # Region of Interest
        roi_ul = (100, img.shape[0])
        roi_ur = (450, 330)
        roi_lr = (520, 330)
        roi_ll = (img.shape[1], img.shape[0])
        vertices = np.array([[roi_ul, roi_ur, roi_lr, roi_ll]], dtype=np.int32)
        target = self.region_of_interest(combined, vertices)
        """

        # Perspective Transform
        img_pt = self.pers_transform.warp(combined, self.file_name, self.data_dir, "_combined")
        img_ud_pt = self.pers_transform.warp(undistorted_img, self.file_name, self.data_dir, "_undistorted")
        # Lane Finding
        lane_img, window_img, histogram, window_fit_img = self.lane_finding.findLines(undistorted_img, img_pt, self.pers_transform, self.file_name, self.data_dir)
        from PIL import Image
        new_width = int(self.getImageSize(img)[0]/4)
        new_height = int(new_width * self.getImageSize(img)[1] / self.getImageSize(img)[0])
        # open the image
        Image1 = Image.fromarray(lane_img)
        # make a copy the image so that the
        # original image does not get affected
        Image1copy = Image1.copy()
        Image2 = Image.fromarray(combined*255).resize((new_width, new_height), Image.ANTIALIAS)
        Image2copy = Image2.copy()

        Image3 = Image.fromarray(img_ud_pt).resize((new_width, new_height), Image.ANTIALIAS)
        Image3copy = Image3.copy()

        Image4 = Image.fromarray(window_img).resize((new_width, new_height), Image.ANTIALIAS)
        Image4copy = Image4.copy()

        # paste image giving dimensions
        Image1copy.paste(Image2copy, (self.getImageSize(img)[0]-new_width, 0))
        Image1copy.paste(Image3copy, (self.getImageSize(img)[0]-new_width, 20 + new_height))
        Image1copy.paste(Image4copy, (self.getImageSize(img)[0]-new_width, 40 + new_height*2))

        Image1_hist = Image.fromarray(img_pt*255)
        Image1hist = Image1_hist.copy()
        Image2_hist = Image.fromarray(histogram).resize((self.getImageSize(img)[0], int(self.getImageSize(img)[1]/2)), Image.ANTIALIAS)
        Image2hist = Image2_hist.copy()
        Image1hist.paste(Image2hist, (0, int(self.getImageSize(img)[1]/2)))

        histogram_image_overlay = np.asarray(Image1hist)

        debug_image = np.asarray(Image1copy)
        # save the image
        if self.save_mode:
            # Save input images
            save_image(img, self.file_name, self.data_dir, "input")
            save_image(undistorted_img, self.file_name, self.data_dir, "undistorted")
            save_image(gradx, self.file_name, self.data_dir, "gradx")
            save_image(grady, self.file_name, self.data_dir, "grady")
            save_image(mag_binary, self.file_name, self.data_dir, "mag_binary")
            save_image(dir_binary, self.file_name, self.data_dir, "dir_binary")
            save_image(h_binary, self.file_name, self.data_dir, "h_binary")
            save_image(l_binary, self.file_name, self.data_dir, "l_binary")
            save_image(s_binary, self.file_name, self.data_dir, "s_binary")
            save_image(_h_binary, self.file_name, self.data_dir, "_h_binary")
            save_image(_s_binary, self.file_name, self.data_dir, "_s_binary")
            save_image(_v_binary, self.file_name, self.data_dir, "_v_binary")
            save_image(color_binary, self.file_name, self.data_dir, "color_binary")
            save_image(combined, self.file_name, self.data_dir, "combined")
            save_image(img_pt, self.file_name, self.data_dir, "perspective")
            save_image(img_ud_pt, self.file_name, self.data_dir, "undistorted_perspective")
            save_image(lane_img, self.file_name, self.data_dir, "lanes")
            save_image(window_img, self.file_name, self.data_dir, "windows")
            save_image(window_fit_img, self.file_name, self.data_dir, "windows_fit")
            save_image(histogram, self.file_name, self.data_dir, "histogram")
            save_image(histogram_image_overlay, self.file_name, self.data_dir, "histogram_overlay")
            save_image(debug_image, self.file_name, self.data_dir, "debug")

        # Increment video frames in case of video processing
        if self.video_mode:
            self.frame_cnt += 1

        return lane_img
