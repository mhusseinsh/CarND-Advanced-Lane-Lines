import numpy as np
import cv2
from src.line import Line, Lane
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class LaneFinding:
    def __init__(self, laneFinding, testImagesMode):
        self.ym_per_pix = eval(laneFinding["ym_per_pix"])
        self.xm_per_pix = eval(laneFinding["xm_per_pix"])
        self.nwindows = int(laneFinding["nwindows"])
        self.margin = int(laneFinding["margin"])
        self.minpix = int(laneFinding["minpix"])

        self.right_line = Line()
        self.left_line = Line()
        self.lane = Lane()

        self.test_images_mode = testImagesMode

    def findLines(self, img, binary_warped, pers_transform, file_name, data_dir):
        # check if a previous fit is detected
        if (self.left_line.detected is False) or (self.right_line.detected is False):
            try:
                print(file_name, "Line Fit")
                left_fit, right_fit, lanes_colored, histogram, window_fit_img = self.lineFit(binary_warped)
                # if nothing was found, use previous fit
            except TypeError:
                left_fit = self.left_line.previous_fit
                right_fit = self.right_line.previous_fit
                lanes_colored = np.zeros_like(img)
        else:
            histogram = img
            window_fit_img = img
            try:
                print(file_name, "Fine Fit")
                left_fit, right_fit, lanes_colored = self.fineFit(binary_warped, self.left_line.previous_fit,
                                                                  self.right_line.previous_fit)
            except TypeError:
                try:
                    left_fit, right_fit, lanes_colored = self.lineFit(binary_warped)
                # if nothing was found, use previous fit
                except TypeError:
                    left_fit = self.left_line.previous_fit
                    right_fit = self.right_line.previous_fit
                    lanes_colored = np.zeros_like(img)

        self.left_line.current_fit = left_fit
        self.right_line.current_fit = right_fit

        # Calculate base position of lane lines to get lane distance
        self.left_line.line_base_pos = left_fit[0] * (binary_warped.shape[0] - 1) ** 2 + left_fit[1] * (
                binary_warped.shape[0] - 1) + left_fit[2]
        self.right_line.line_base_pos = right_fit[0] * (binary_warped.shape[0] - 1) ** 2 + right_fit[1] * (
                binary_warped.shape[0] - 1) + right_fit[2]
        self.left_line.line_mid_pos = left_fit[0] * (binary_warped.shape[0] // 2) ** 2 + left_fit[1] * (
                binary_warped.shape[0] // 2) + left_fit[2]
        self.right_line.line_mid_pos = right_fit[0] * (binary_warped.shape[0] // 2) ** 2 + right_fit[1] * (
                binary_warped.shape[0] // 2) + right_fit[2]

        # Calculate top and bottom position of lane lines for sanity check
        self.lane.top_width = right_fit[2] - left_fit[2]
        self.lane.bottom_width = self.right_line.line_base_pos - self.left_line.line_base_pos
        self.lane.middle_width = self.right_line.line_mid_pos - self.left_line.line_mid_pos

        # Check if values make sense
        if self.sanity_check() is False:
            # If fit is not good, use previous values and indicate that lanes were not found
            if len(self.left_line.previous_fits) == 5:
                diff_left = [0.0, 0.0, 0.0]
                diff_right = [0.0, 0.0, 0.0]
                for i in range(0, 3):
                    for j in range(0, 3):
                        diff_left[i] += self.left_line.previous_fits[j][i] - self.left_line.previous_fits[j + 1][i]
                        diff_right[i] += self.right_line.previous_fits[j][i] - self.right_line.previous_fits[j + 1][i]

                    diff_left[i] /= 4
                    diff_right[i] /= 4

                for i in range(0, 3):
                    self.left_line.current_fit[i] = self.left_line.previous_fit[i] + diff_left[i]
                    self.right_line.current_fit[i] = self.right_line.previous_fit[i] + diff_right[i]
                print("prev: ", self.left_line.previous_fit)
                print("diff: ", diff_left)
                print("fit: ", self.left_line.current_fit)

                self.left_line.detected = False
                self.right_line.detected = False
            else:
                self.left_line.current_fit = self.left_line.previous_fit
                self.right_line.current_fit = self.right_line.previous_fit
                self.left_line.detected = False
                self.right_line.detected = False

        else:
            # If fit is good, use current values and indicate that lanes were found
            if not self.left_line.detected or not self.right_line.detected:
                self.left_line.previous_fits.clear()
                self.right_line.previous_fits.clear()
            self.left_line.detected = True
            self.right_line.detected = True
            self.left_line.initialized = True
            self.right_line.initialized = True
            self.left_line.frame_cnt += 1
            self.right_line.frame_cnt += 1

        # Calculate the average of the recent fits and set this as the current fit
        self.left_line.average_fit = self.average_fits(binary_warped.shape, self.left_line)
        self.right_line.average_fit = self.average_fits(binary_warped.shape, self.right_line)

        self.lane.average_bottom_width, self.lane.average_top_width = self.average_width(binary_warped.shape)

        # Determine lane curvature and position of the vehicle
        self.left_line.radius_of_curvature = self.calc_curvature(binary_warped.shape, left_fit)
        self.right_line.radius_of_curvature = self.calc_curvature(binary_warped.shape, right_fit)
        curvature = self.left_line.radius_of_curvature + self.right_line.radius_of_curvature / 2

        self.left_line.line_base_pos = left_fit[0] * (binary_warped.shape[0] - 1) ** 2 + left_fit[1] * (
                binary_warped.shape[0] - 1) + left_fit[2]
        self.right_line.line_base_pos = right_fit[0] * (binary_warped.shape[0] - 1) ** 2 + right_fit[1] * (
                binary_warped.shape[0] - 1) + right_fit[2]
        vehicle_position = self.vehicle_position(binary_warped.shape[1], self.left_line.line_base_pos,
                                                 self.right_line.line_base_pos)

        # Warp lane boundaries back & display lane boundaries, curvature and position
        lanes_marked = self.visualize_lines(binary_warped, img, self.left_line.average_fit, self.right_line.average_fit,
                                            curvature, vehicle_position, pers_transform, file_name, data_dir)

        # Set current values as previous values for next frame
        self.left_line.previous_fit = self.left_line.current_fit
        self.right_line.previous_fit = self.right_line.current_fit

        # Reset / empty current fit
        self.left_line.current_fit = [np.array([False])]
        self.right_line.current_fit = [np.array([False])]

        if self.test_images_mode:
            self.left_line.reset()
            self.right_line.reset()

        return lanes_marked, lanes_colored.astype("uint8"), histogram, window_fit_img

    def lineFit(self, binary_warped):
        """
        Find and fit lane lines
        """
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        bottom_half = binary_warped[binary_warped.shape[0] // 2:, :]
        histogram = np.sum(bottom_half, axis=0)

        # Output image for testing
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Create starting point on left and right side and set them as current points
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Parameters of sliding window
        # Number of windows
        nwindows = self.nwindows
        # Width of windows
        margin = self.margin
        # Minimum number of pixels to recenter window
        minpix = self.minpix

        # Set window height
        window_height = np.int(binary_warped.shape[0] // nwindows)

        # Find nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        # Empty lists for storing lane pixel indices
        left_line_inds = []
        right_line_inds = []

        # Step through windows
        for window in range(nwindows):

            # Window boundaries:
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify pixels within the windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Add the found pixels to lane line
            left_line_inds.append(good_left_inds)
            right_line_inds.append(good_right_inds)

            # Update x axis position based on pixels found
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate list of pixels
        try:
            left_line_inds = np.concatenate(left_line_inds)
            right_line_inds = np.concatenate(right_line_inds)
        except ValueError:
            pass

        # Get left and right lane pixel positions
        leftx = nonzerox[left_line_inds]
        lefty = nonzeroy[left_line_inds]
        rightx = nonzerox[right_line_inds]
        righty = nonzeroy[right_line_inds]

        # Color left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Fit polynomial based on pixels found
        left_fit, right_fit = self.fit_poly(leftx, lefty, rightx, righty)

        # Draw Histogram
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.plot(histogram, color="b")
        canvas.draw()
        buf = canvas.buffer_rgba()
        histogram_img = np.asarray(buf)

        # Draw Fit line
        left_fitx, right_fitx = self.calc_x_values(binary_warped.shape, left_fit, right_fit)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.imshow(out_img.astype("uint8"), aspect='auto')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        canvas.draw()
        buf = canvas.buffer_rgba()
        window_fit_img = np.asarray(buf)


        return left_fit, right_fit, out_img, histogram_img, window_fit_img

    def fineFit(self, binary_warped, left_fit, right_fit):
        """
        Given a previously fit line, quickly try to find the line based on previous lines
        """
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        # get activated pixels
        # Margin for searching around curve
        margin = self.margin

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Mask to get non-zero pixels that are next to the curve within margin
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Generate y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # Fit polynomial based on pixels found
        left_fit, right_fit = self.fit_poly(leftx, lefty, rightx, righty)

        left_fitx, right_fitx = self.calc_x_values(binary_warped.shape, left_fit, right_fit)

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return left_fit, right_fit, out_img

    def fit_poly(self, leftx, lefty, rightx, righty):

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def sanity_check(self):
        # Calculate widths at top and bottom
        top_width_diff = abs(self.lane.top_width - self.lane.average_top_width)
        bottom_width_diff = abs(self.lane.bottom_width - self.lane.average_bottom_width)

        # Define sanity checks
        width_check_top = top_width_diff > 0.2 * self.lane.average_top_width or self.lane.top_width > 1.25 * self.lane.bottom_width
        width_check_bottom = bottom_width_diff > 0.05 * self.lane.average_bottom_width
        lane_intersect_check = self.lane.top_width < 0.0 or self.lane.bottom_width < 0.0
        curve_check = self.right_line.current_fit[0] * self.left_line.current_fit[0] < -0.00005 * 0.0001

        # Check if parameters are ok (skip for first frame)
        if (self.left_line.frame_cnt > 1) and (self.right_line.frame_cnt > 1):
            if width_check_bottom:
                result = False
            elif width_check_top:
                result = False
            elif lane_intersect_check:
                result = False
            elif curve_check:
                result = False
            else:
                result = True
        else:
            result = True

        return result

    def average_fits(self, img_shape, line):
        n = 3
        average_fit = [0, 0, 0]

        # Append the previous fits in case of there are no fits stored
        if len(line.previous_fits) < n:
            line.previous_fits.append(line.current_fit)
        # If list is full, replace the first fit with the current one
        if len(line.previous_fits) == n:
            line.previous_fits.pop(n - 1)
            line.previous_fits.insert(0, line.current_fit)

        # Average fit
        if len(line.previous_fits) > 0:
            for i in range(0, 3):
                total = 0
                for num in range(0, len(line.previous_fits)):
                    total = total + line.previous_fits[num][i]

                average_fit[i] = total / len(line.previous_fits)

        return average_fit

    def average_width(self, img_shape):
        sum_bottom = 0
        sum_top = 0
        n = 3
        average_bottom_width = 0
        average_top_width = 0

        if len(self.lane.previous_bottom_widths) < n:
            self.lane.previous_bottom_widths.append(self.lane.bottom_width)
        # If list is full, replace the first fit with the current one
        if len(self.lane.previous_bottom_widths) == n:
            self.lane.previous_bottom_widths.pop(n - 1)
            self.lane.previous_bottom_widths.insert(0, self.lane.bottom_width)

        # Average width
        if (len(self.lane.previous_bottom_widths) > 0):
            for i in range(0, len(self.lane.previous_bottom_widths)):
                sum_bottom = sum_bottom + self.lane.previous_bottom_widths[i]
                average_bottom_width = sum_bottom / len(self.lane.previous_bottom_widths)

        if len(self.lane.previous_top_widths) < n:
            self.lane.previous_top_widths.append(self.lane.top_width)
        # If list is full, replace the first fit with the current one
        if len(self.lane.previous_top_widths) == n:
            self.lane.previous_top_widths.pop(n - 1)
            self.lane.previous_top_widths.insert(0, self.lane.top_width)

        # Average width
        if len(self.lane.previous_top_widths) > 0:
            for i in range(0, len(self.lane.previous_top_widths)):
                sum_top = sum_top + self.lane.previous_top_widths[i]
                average_top_width = sum_top / len(self.lane.previous_top_widths)

        return average_bottom_width, average_top_width

    def visualize_lines(self, warped, undist, left_fit, right_fit, curvature, position, persp_transform, file_name,
                        data_dir):
        # Generate y values
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

        # Calculate x values
        left_fitx, right_fitx = self.calc_x_values(warped.shape, left_fit, right_fit)

        # Create image to draw lines onto
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.array([pts_left], dtype=np.int32), False, (255, 0, 0), thickness=15)
        cv2.polylines(color_warp, np.array([pts_right], dtype=np.int32), False, (0, 0, 255), thickness=15)
        # Warp the blank back to original image space
        newwarp = persp_transform.warpInv(color_warp, file_name, data_dir, "_lanes")

        # Combine the result with the original image
        lanes = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Texts to write on image
        curv_text = "Radius of Curvature: %.2f meters" % curvature
        if position >= 0:
            pos_text = "Position: %.2f right from center" % position
        else:
            pos_text = "Position: %.2f left from center" % abs(position)

        # Add text to image
        cv2.putText(lanes, curv_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lanes, pos_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return lanes

    def calc_curvature(self, img_shape, fit):
        # Generate y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        # Calculate x values using polynomial coeffs
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

        # Evaluate at bottom of image
        y_eval = np.max(ploty)

        # Fit curves with corrected axes
        curve_fit = np.polyfit(ploty * self.ym_per_pix, fitx * self.xm_per_pix, 2)

        # Calculate curvature values for left and right lanes
        curvature = ((1 + (2 * curve_fit[0] * y_eval * self.ym_per_pix + curve_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * curve_fit[0])

        return curvature

    def vehicle_position(self, img_shape, left_lane_pos, right_lane_pos):
        # Calculate position based on midpoint - center of lanes distance
        midpoint = img_shape // 2
        center_of_lanes = (right_lane_pos + left_lane_pos) / 2
        position = midpoint - center_of_lanes

        # Get value in meters
        real_position = position * self.xm_per_pix

        return real_position

    def calc_x_values(self, img_shape, left_fit, right_fit):
        # Generate y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        # Calculate x values using polynomial coeffs
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx
