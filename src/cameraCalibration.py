import numpy as np
import cv2
import glob
import pickle
import os
from src.utils import check_dir, save_image
import time
import sys


class CameraCalibration:

    def __init__(self, chessboardSize, path):
        self.path = path
        self.showImages = False
        self.chessboardSize = chessboardSize
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    def calibrateCamera(self):
        calibration_results = os.path.join(self.path + "_results", "calibration_results_pickle.p")
        if os.path.exists(calibration_results):
            print("calibration results file exists, loading from file.")
            ret, mtx, dist, rvecs, tvecs = self.calibrateCameraFromFile(calibration_results)
        else:
            print("calibration results file does not exist, start calibration from scratch.")
            ret, mtx, dist, rvecs, tvecs = self.calibrateCameraFromImages(self.path)
            # Only to have a copy for the undistorted chessboard images
            from src.undistortion import Undistortion
            ud = Undistortion(mtx, dist)
            ud.undistortChessboardImages(self.path)
        return ret, mtx, dist, rvecs, tvecs

    def calibrateCameraFromImages(self, path):
        print("Calibrating:")
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.
        images = glob.glob(path + "/*")

        # setup toolbar
        toolbar_width = len(images)
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
        # Step through the list and search for chessboard corners
        for idx, file_name in enumerate(images):
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, None)

            # If found, add object points, image points
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)  # Draw and display the corners
                cv2.drawChessboardCorners(img, self.chessboardSize, corners, ret)
                # save image
                save_image(img, file_name, path, "corners")
                if self.showImages:
                    cv2.imshow('img', img)
                    cv2.waitKey(500)
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()
        sys.stdout.write("]\n")  # this ends the progress bar
        print("Calibration done. Saving calibration results.")
        if self.showImages:
            cv2.destroyAllWindows()
        # Do camera calibration given object points and image points
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
        calibration_file = os.path.join(self.path + "_results", "calibration_results_pickle.p")
        check_dir(calibration_file)
        pickle.dump(dist_pickle, open(calibration_file, "wb"))
        return ret, mtx, dist, rvecs, tvecs

    def calibrateCameraFromFile(self, calibration_results):
        with open(calibration_results, "rb") as f:
            dist_pickle = pickle.load(f)
        return dist_pickle["ret"], dist_pickle["mtx"], dist_pickle["dist"], dist_pickle["rvecs"], dist_pickle[
            "tvecs"]
