import cv2
import numpy as np
from src.utils import my_float


def colorspace_select(img, thresh=(0, 255), colorspace="hls", channel="s_channel"):
    if colorspace == "hls":
        channels = hls_select(img)
    elif colorspace == "hsv":
        channels = hsv_select(img)
    binary_output = np.zeros_like(channels[channel])
    binary_output[(channels[channel] > thresh[0]) & (channels[channel] <= thresh[1])] = 1
    return binary_output

def hls_select(img):
    channels = {}
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channels["h_channel"] = hls[:, :, 0]
    channels["l_channel"] = hls[:, :, 1]
    channels["s_channel"] = hls[:, :, 2]
    return channels

def hsv_select(img):
    channels = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channels["h_channel"] = hsv[:, :, 0]
    channels["s_channel"] = hsv[:, :, 1]
    channels["v_channel"] = hsv[:, :, 2]
    return channels
