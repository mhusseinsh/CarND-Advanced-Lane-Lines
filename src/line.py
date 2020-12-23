import numpy as np


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.previous_fits = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Frame counter
        self.frame_cnt = 0
        # Average fit
        self.average_fit = np.array([0, 0, 0])


    def reset(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.previous_fits = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Frame counter
        self.frame_cnt = 0
        # Average fit
        self.average_fit = np.array([0, 0, 0])

class Lane:
    def __init__(self):
        self.bottom_width = 0
        self.top_width = 0
        self.average_bottom_width = 0
        self.average_top_width = 0
        self.previous_bottom_widths = []
        self.previous_top_widths = []