from src.utils import save_image
import cv2
import os
import glob
import matplotlib.image as mpimg


class Undistortion:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def undistort(self, data_dir, file_name, img):
        undistorted_img = cv2.undistort(img, self.mtx, self.dist)
        return undistorted_img

    def undistortChessboardImages(self, data_dir):
        images = glob.glob(data_dir + "/*")
        for idx, file_name in enumerate(images):
            img = mpimg.imread(file_name)
            undistorted_img = self.undistort(data_dir, file_name, img)
            save_image(undistorted_img, file_name, data_dir, "undistorted")
