from src.utils import save_image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PerspectiveTransform:
    def __init__(self, perspectiveTransform):
        # ul, ll, lr, ur
        src_offsets = perspectiveTransform["src"]
        self.ul_offset = [int(src_offsets["ul_offset"]["x"]), int(src_offsets["ul_offset"]["y"])]
        self.ll_offset = [int(src_offsets["ll_offset"]["x"]), int(src_offsets["ll_offset"]["y"])]
        self.lr_offset = [int(src_offsets["lr_offset"]["x"]), int(src_offsets["lr_offset"]["y"])]
        self.ur_offset = [int(src_offsets["ur_offset"]["x"]), int(src_offsets["ur_offset"]["y"])]

    def getCornerPoints(self, img_size):
        src = np.float32(
            [[(img_size[0] / 2 + self.ul_offset[0]), img_size[1] / 2 + self.ul_offset[1]],
             [(img_size[0] / 2 + self.ll_offset[0]), img_size[1] + self.ll_offset[1]],
             [(img_size[0] / 2 + self.lr_offset[0]), img_size[1] + self.lr_offset[1]],
             [(img_size[0] / 2 + self.ur_offset[0]), img_size[1] / 2 + self.ur_offset[1]]])

        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

        return src, dst

    def warp(self, img, file_name, data_dir):
        img_size = img.shape[1], img.shape[0]
        src, dst = self.getCornerPoints(img_size)
        M = cv2.getPerspectiveTransform(src, dst)
        perpImage = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        self.saveSideImages(img, perpImage, file_name, data_dir, src, dst)
        return perpImage

    def warpInv(self, img, file_name, data_dir):
        img_size = img.shape[1], img.shape[0]
        src, dst = self.getCornerPoints(img_size)
        M = cv2.getPerspectiveTransform(dst, src)
        perpImage = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        self.saveSideImages(perpImage, img, file_name, data_dir, src, dst)
        return perpImage

    def saveSideImages(self, img, perpImage, file_name, data_dir, src, dst):
        f, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Thresholded Image with source points drawn')
        ax[1].imshow(perpImage, cmap='gray')
        ax[1].set_title('Warped result with dest. points drawn')
        ax[0].axis('off')
        ax[1].axis('off')
        x = [src[0][0], src[1][0], src[2][0], src[3][0], src[0][0]]
        y = [src[0][1], src[1][1], src[2][1], src[3][1], src[0][1]]
        x_ = [dst[0][0], dst[1][0], dst[2][0], dst[3][0], dst[0][0]]
        y_ = [dst[0][1], dst[1][1], dst[2][1], dst[3][1], dst[0][1]]
        ax[0].plot(x, y, 'b--', lw=2)
        ax[1].plot(x_, y_, 'b--', lw=2)
        f.tight_layout()
        f.savefig("test.jpg")
        save_image(mpimg.imread("test.jpg"), file_name, data_dir, "perpSide")
        os.remove("test.jpg")
