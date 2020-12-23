## Report
---
![alt text][project_video]

**Advanced Lane Finding Project**

Lane finding is an important task for an autonomous vehicle to be able to how much exactly is it away from a certain lane. Finding the ego lane as well and making sure that the car is sticking to it is very crucial for self-driving vehicles. Lane detection may be tricky sometimes, especially in the case of changing of light/brightness or different lane colors as well as some road markings may be disappearing. Previously, a [basic lane finding algorithm](https://github.com/mhusseinsh/CarND-LaneLines-P1) using classic approaches of basic [openCV](https://opencv.org/) functions was implemented, and it was very obvious from the results, that it cannot be used for generic cases. The fact is due that it failed in curvy lanes, and different lane colors.

The aim of this project was to implement an advanced lane finding algorithm which can be used in most of the cases and can easily find the lanes for different scenarios.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[camera_cal_input]: ./camera_cal/calibration2.jpg "Chessboard Input"
[camera_cal_corners]: ./output_images/camera_cal_corners/calibration2_corners.jpg "Chessboard Corners"
[camera_cal_undistorted]: ./output_images/camera_cal_undistorted/calibration2_undistorted.jpg "Chessboard Undistorted"

[test_image_input]: ./test_images/test2.jpg
[test_image_undistorted]: ./output_images/test_images_undistorted/test2_undistorted.jpg
[test_image_gradx]: ./output_images/test_images_gradx/test2_gradx.jpg
[test_image_grady]: ./output_images/test_images_grady/test2_grady.jpg
[test_image_mag_binary]: ./output_images/test_images_mag_binary/test2_mag_binary.jpg
[test_image_dir_binary]: ./output_images/test_images_dir_binary/test2_dir_binary.jpg
[test_image_h_binary]: ./output_images/test_images_h_binary/test2_h_binary.jpg
[test_image_l_binary]: ./output_images/test_images_l_binary/test2_l_binary.jpg
[test_image_s_binary]: ./output_images/test_images_s_binary/test2_s_binary.jpg
[test_image__h_binary]: ./output_images/test_images__h_binary/test2__h_binary.jpg
[test_image__s_binary]: ./output_images/test_images__s_binary/test2__s_binary.jpg
[test_image__v_binary]: ./output_images/test_images__v_binary/test2__v_binary.jpg
[test_image_combined]: ./output_images/test_images_combined/test2_combined.jpg
[test_image_perspective]: ./output_images/test_images_perspective/test2_perspective.jpg
[test_image_undistorted_perspective]: ./output_images/test_images_undistorted_perspective/test2_undistorted_perspective.jpg
[test_image_lanes]: ./output_images/test_images_lanes/test2_lanes.jpg
[test_image_windows]: ./output_images/test_images_windows/test2_windows.jpg
[test_image_windows_2]: ./output_images/test_images_windows/test2_copy_windows.jpg
[test_image_windows_fit]: ./output_images/test_images_windows_fit/test2_windows_fit.jpg
[test_image_histogram]: ./output_images/test_images_histogram/test2_histogram.jpg
[side_by_side_straight]: ./output_images/test_images_perpSide_undistorted/straight_lines1_perpSide_undistorted.jpg
[side_by_side_straight_combined]: ./output_images/test_images_perpSide_combined/straight_lines1_perpSide_combined.jpg
[side_by_side_curved]: ./output_images/test_images_perpSide_undistorted/test2_perpSide_undistorted.jpg
[side_by_side_curved_combined]: ./output_images/test_images_perpSide_combined/test2_perpSide_combined.jpg
[project_video]: ./output_videos/project_video_output.gif
[challenge_video]: ./output_videos/challenge_video_output.gif
[harder_challenge_video]: ./output_videos/harder_challenge_video_output.gif

[project_video_debug]: ./output_videos/project_video_output_debug.gif
[challenge_video_debug]: ./output_videos/challenge_video_output_debug.gif
[harder_challenge_video_debug]: ./output_videos/harder_challenge_video_output_debug.gif


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Camera calibration is the process of finding the true parameters of the camera model we are working with. In autonomous driving, this is essential to ensure that image measurements result in accurate estimates of locations and dimensions within the object space.

As a first step, camera calibration is conducted. This used some chessboard images of size <strong>(9*6)</strong> that were provided [here](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/tree/master/camera_cal). Below are some examples of the chessboard images:

<img src="./camera_cal/calibration6.jpg" width="200"/> <img src="./camera_cal/calibration7.jpg" width="200"/>  <img src="./camera_cal/calibration8.jpg" width="200"/> <img src="./camera_cal/calibration9.jpg" width="200"/> <img src="./camera_cal/calibration10.jpg" width="200"/>  <img src="./camera_cal/calibration11.jpg" width="200"/> <img src="./camera_cal/calibration12.jpg" width="200"/> <img src="./camera_cal/calibration13.jpg" width="200"/> 


A [`class CameraCalibration`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/cameraCalibration.py#L11) which contains some class members and the [`calibrateCameraFromImages`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/cameraCalibration.py#L35) function is defined in [cameraCalibration.py](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/master/src/cameraCalibration.py).

The calibration process starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. 
```python
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
self.objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
self.objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
```
Thus, `objp` is just a replicated array of coordinates like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0), and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. This is achieved using the the [`cv2.findChessboardCorners()`](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a) function.
```python
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
```
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the [`cv2.calibrateCamera()`](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) function.  I applied this distortion correction to the test image using the [`cv2.undistort()`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/undistortion.py#L17) function, which will be explained later (just for debugging)
```python
# Do camera calibration given object points and image points
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
```
The obtained results:

Chessboard | Detected Corners | Undistorted Chessboard
:-:|:-:|:-:
![alt text][camera_cal_input] | ![alt text][camera_cal_corners] | ![alt text][camera_cal_undistorted]

After obtaining the camera calibration coefficients, they are saved locally, so they can be called again using the [`calibrateCameraFromFile`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/cameraCalibration.py#L80) function, which saves time and effort for rerunning the calibration every time for any new test image.
```python
def calibrateCameraFromFile(self, calibration_results):
    with open(calibration_results, "rb") as f:
        dist_pickle = pickle.load(f)
    return dist_pickle["ret"], dist_pickle["mtx"], dist_pickle["dist"], dist_pickle["rvecs"], dist_pickle[
        "tvecs"]
```
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The complete pipeline begins by running the calibration once (if the calibration parameters are not found locally), then getting the coefficients, and start running on a single input image. The image is first undistorted using the [`undistort`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/undistortion.py#L13) which is defined in the [`class Undistortion`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/undistortion.py#L8) in [undistortion.py](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/master/src/undistortion.py#L8). Inside this function, the camera calibration matrix and distortion coefficients are passed to the [`cv2.undistort()`](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d) along with the input distorted image.
```python
def undistort(self, data_dir, file_name, img):
    undistorted_img = cv2.undistort(img, self.mtx, self.dist)
    return undistorted_img
```
The output image is returned undistorted.
Input Image | Undistorted Image
:-:|:-:
![alt text][test_image_input] | ![alt text][test_image_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A combination of color and gradient thresholds is then used to generate a binary image. The undistorted image is used to get the Gradient Threshold, Magnitued of the Gradient, Direction of the Gradient and finally a threshold color channel in the HLS space.

The thresholding functions are defined in [gradientThresh.py](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/master/src/gradientThresh.py) and [colorSpaces.py](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/master/src/colorSpaces.py) files.
* <strong>Gradient Threshold</strong>
    
    The directional gradient for both <em>x</em> and <em>y</em> is calculated using the below function.
    ```python
    # Calculate directional gradient
    # Apply threshold
    def abs_sobel_threshold(img, orient='x', sobel_kernel=3, grad_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate x or y directional gradient
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        if orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take absolute sobel value
        abs_sobel = np.absolute(sobel)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1

        # Return the result
        return binary_output
    ```
    This produces a binary image for both <em>x</em> and <em>y</em> directions as below
    gradx | grady
    :-:|:-:
    ![alt text][test_image_gradx] | ![alt text][test_image_grady]

* <strong>Gradient Magnitude</strong>
    
    The magnitude of the gradient is calculated using the below function.
    ```python
    # Calculate gradient magnitude
    # Apply threshold
    def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(abs_sobelxy) / 255
        gradmag = (abs_sobelxy / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output
    ```
    This produces the below binary image
    ![alt text][test_image_mag_binary]

* <strong>Gradient Direction</strong>
    
    The direction of the gradient is calculated using the below function.
    ```python
    # Calculate gradient direction
    # Apply threshold
    def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi / 2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # Calculate Direction
        direction = np.arctan2(abs_sobely, abs_sobelx)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= dir_thresh[0]) & (direction <= dir_thresh[1])] = 1

        # Return the binary image
        return binary_output
    ```
    This produces the below binary image
    ![alt text][test_image_dir_binary]

* <strong>Color Channel - HLS and HSV Spaces</strong>
    
    The color channel thresholds are calculated using the below function.
    ```python
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
    ```
    This produces the below binary images in HLS Space
    h-channel | l-channel | s-channel
    :-:|:-:|:-:
    ![alt text][test_image_h_binary] | ![alt text][test_image_l_binary] | ![alt text][test_image_s_binary]

    And the below binary images in HSV Space
    h-channel | s-channel | v-channel
    :-:|:-:|:-:
    ![alt text][test_image__h_binary] | ![alt text][test_image__s_binary] | ![alt text][test_image__v_binary]

    It is clearly obvious that the s-channel outperforms in both HLS and HSV spaces as it shows the lines very well.
    
* Combined Binary Image
  
  The color and gradient thresholds are combined to be able to achieve the best both worlds.
  ![alt text][test_image_combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

A perspective transform maps the points in a given image to different, desired, image points with a new perspective. In this case, we are interested in the bird’s-eye view transform, which allows us to view the lane from above, making it easier to later calculate the lane curvature and so on. A [`class PerspectiveTransform`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/perspectiveTransform.py#L9) is defined in [perspectiveTransform.py](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/tree/master/src) which contains some class members from predefined corner offsets that are written in a configurations file for the `src` and `dst` corner points.
```python
def __init__(self, perspectiveTransform):
    # ul, ll, lr, ur
    src_offsets = perspectiveTransform["src"]
    self.ul_offset = [int(src_offsets["ul_offset"]["x"]), int(src_offsets["ul_offset"]["y"])]
    self.ll_offset = [int(src_offsets["ll_offset"]["x"]), int(src_offsets["ll_offset"]["y"])]
    self.lr_offset = [int(src_offsets["lr_offset"]["x"]), int(src_offsets["lr_offset"]["y"])]
    self.ur_offset = [int(src_offsets["ur_offset"]["x"]), int(src_offsets["ur_offset"]["y"])]
```
Then the `src` and `dst` points are calculated based on the below formulas
```python
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
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 610, 440      | 320, 0        | 
| 220, 720      | 320, 720      |
| 1190, 720     | 720, 720      |
| 700, 440      | 720, 0        |

The `class` includes a [`warp`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/perspectiveTransform.py#L33) and [`warpInv`](https://github.com/mhusseinsh/CarND-Advanced-Lane-Lines/blob/d1a0097700b232b67bba15b278791e643ab9ec9a/src/perspectiveTransform.py#L41) functions which call inside the [`cv2.getPerspectiveTransform()`](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae) and [`cv2.warpPerspective()`](https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87) respectively to get the transformation matrix `M` between two arrays then apply it on an input image to get the transformed one.
```python
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
```
Input Image | Warped Image
:-:|:-:
![alt text][test_image_undistorted] | ![alt text][test_image_undistorted_perspective]

Thresholded Image | Warped Thresholded Image
:-:|:-:
![alt text][test_image_combined] | ![alt text][test_image_perspective]

The perspective transform was verified by drawing the `src` and `dst` points on test images and their transformations to verify that the straight lines appear parallel in the warped image and also showing that the curved lines are (more or less) parallel in the transformed image.


![alt text][side_by_side_straight]
![alt text][side_by_side_straight_combined]
![alt text][side_by_side_curved]
![alt text][side_by_side_curved_combined]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and a perspective transform to a road image, A binary image where the lane lines stand out clearly is now available. However, it is still needed to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.
Input Image | Warped Thresholded Image
:-:|:-:
![alt text][test_image_undistorted] | ![alt text][test_image_perspective]

Plotting a histogram of where the binary activations occur across the image could be a first step to do.
```python
# Take a histogram of the bottom half of the image
bottom_half = binary_warped[binary_warped.shape[0] // 2:, :]
histogram = np.sum(bottom_half, axis=0)
```
![alt text][test_image_histogram]

With this histogram above, the pixel values along each column in the image are added up. If we check back the thresholded binary image, we have pixels only containing 0 or 1, so the two most prominent peaks in this histogram show the x-position of the base of the lane lines. From this base, a lane search algorithm can be imlemented.

From this step, a sliding window search algorithm is implemented. The idea is to simply create windows along the height of the image (for both left and right lines) and start moving these windows upward in the image (further along the road) to determine where the lane lines go. The windows should keep iterating across the binary activations in the image to track curvature with sliding left or right if it finds the mean position of activated pixels within the window to have shifted.
![alt text][test_image_windows]

After getting all the pixels belonging the left and right lanes, we fit a polynomial through these pixels to find the left and right fit using the [`np.polyfit()`](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html) function.
![alt text][test_image_windows_fit]

This is implemented in the [`lineFit`]() function which is a method of the complete [`class LaneFinding`]() implemented in [laneFinding.py]()
```python
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
    # Output values
    #left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
    #right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])

    # Add text to image
    #cv2.putText(out_img, left_fit_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)
    #cv2.putText(out_img, right_fit_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

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
```
Using the same algorithm of sliding window search for each and every frame may seem to be quite inreasonable and inefficient. If we are working with a camera which provides around a real-time FPS, we can notice that between every consequtive frame, there is no such different in the lane lines, in terms of position or curvature. Accordingly, a [`fineFit`]() function is implemented under the same class, which searches for the lanes within a search margin based on the previous saved fit. The idea is that once we use the sliding windows to find the lines fit, the fit is saved. Then for the next frame, we don't search from scratch.
```python
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

    # Output values
    #left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
    #right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])

    # Add text to image
    #cv2.putText(out_img, left_fit_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)
    #cv2.putText(out_img, right_fit_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)

    return left_fit, right_fit, out_img
```
Line Fit (Sliding Window) | Fine Fit (Search around Poly)
:-:|:-:
![alt text][test_image_windows] | ![alt text][test_image_windows_2]

After every algorithm run, a sanity check is done. Inside the [`sanity_check`](), the lines are checked against each other if they do have more or less a similar curvature, if they have more or less the same width in top, middle and bottom positions and if they are roughly parallel. If the sanity check fails, then this is considered to be an invalid fit, and accordingly, the fit is cleared and the line is saved as undetected. This will help that the next frame will do a blind search from scratch instead of depending on the previous detected fit. In this case, a complete reset is done.
```python
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
```
Moreover, an average smoothing of the recent fits is done. This simply tries to prevents the line detections which jump around from frame to frame a bit. So the last 3 detections are averaged to achieve an average fit, and whenever a new successful detection is done, it replaces the very first detection from the 3 saved detections.
```python
def average_fits(self, img_shape, line):
    n = 3
    average_fit = [0, 0, 0]

    # If we do not have enough fits, append the list with the current fit
    if len(line.previous_fits) < n:
        line.previous_fits.append(line.current_fit)
    # If amount of fits == n, remove the last element and add the current one
    if len(line.previous_fits) == n:
        line.previous_fits.pop(n - 1)
        line.previous_fits.insert(0, line.current_fit)

    # If we have enough fits, calculate the average
    if len(line.previous_fits) > 0:
        for i in range(0, 3):
            total = 0
            for num in range(0, len(line.previous_fits)):
                total = total + line.previous_fits[num][i]

            average_fit[i] = total / len(line.previous_fits)

    return average_fit
```
In order to easily achieve all the above work, a [`class Line`]() is defined in [line.py]() which stores all the information about a certain line that can be easily recalled and modified.
```python
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
```
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After we got the lanes, it is time now to do some nice calculations to retrieve more information.

* <strong>Radius of Curvature</strong>
    
    To get the formula of calculating the radius of a curvature, this [reference](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) was used. For calculating a radius of curvature in real world, the U.S. regulations were used. It is required that a lane should have a minimum width of 3.7 meters, and it is assumed the lane's length is about 30m. Therefore, to convert from pixels to real-world meter measurements, we can use:
    ```python
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ```
    Accordingly, the radius of curvature is implemented in the [`calc_curvature()`]() function as below:
    ```python
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
    ```
* <strong>Vehicle Deviation from Center</strong>

    The deviation of the driving vehicle from the center is easily achieved via getting the center position of the image and the center of the detected lane then comparing them together. This is implemented in the [`vehicle_position()`]() function as below:
    ```python
    def vehicle_position(self, img_shape, left_lane_pos, right_lane_pos):
        # Calculate position based on midpoint - center of lanes distance
        midpoint = img_shape // 2
        print(midpoint)
        exit()
        center_of_lanes = (right_lane_pos + left_lane_pos) / 2
        position = midpoint - center_of_lanes

        # Get value in meters
        real_position = position * self.xm_per_pix

        return real_position
    ```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final step of the pipeline was to plot the polynomials on the warped image, fill the space between the polynomials to highlight the ego lane, use inverse perspective trasformation to unwarp the image from the birds-eye view back to its original perspective, and print the distance from center and radius of curvature on to the final annotated image.
![alt text][test_image_lanes]

This is implemented in the [`visualize_lines()`]() function as below:
```python
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
```
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
After establishing a pipeline to process standalone test images, the final step was to make this pipeline to process videos frame-by-frame, to see the performace of camera input stream mounted on a real vehicle.

My goal was not to create a different pipeline or method for video processing that will be much different from image processing. So I tried to stick to using the exact same methods exactly, but just using some flags to differentiate between if it is a video or single images. The [`class Line`]() helped a lot in this process as I mentioned above. I was simply saving all the related line information in terms of previous or current detections, as well as all lane features, which can be used to smooth my detections across all frames.

The check of whether to use the [`lineFit`]() or [`fineFit`]() as explained above was implemented, so the pipeline does not need to scan the whole entire frame once again if there is already a detected line in the previous frame, nevertheless, it searches through a margin around the previous detected line for the location of the new line. 

If for any reason, the [`fineFit`]() fails to get new lane pixels, the pipeline shifts automatically to [`lineFit`]() to do a blind search via the sliding window approach.

The output of the project video is shown below, and here's a [link to the video result](./output_videos/project_video_output.mp4):
|Project Video|Project Video - Debug|
|-------------|-------------|
|![Project Video][project_video]|![Project Video - Debug][project_video_debug]
Another test was done on the challenge video as shown below, and here's a [link to the video result](./output_videos/challenge_video_output.mp4):
|Challenge Video|Challenge Video - Debug|
|-------------|-------------|
|![Challenge Video][challenge_video]|![Challenge Video - Debug][challenge_video_debug]
And a last test was done on the harder challenge video as shown below, and here's a [link to the video result](./output_videos/harder_challenge_video_output.mp4):
|Harder Challenge Video|Harder Challenge Video - Debug|
|-------------|-------------|
|![Harder Challenge Video][harder_challenge_video]|![Harder Challenge Video - Debug][harder_challenge_video_debug]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Summary

    Python3 with openCV functions were used to perform the camera calibration and do the undistortion of an image. Afterwards, focusing on a specific region and applying a perspective tranformation to retrieve a "bird's eye view" of the road. Then using Gradient and Color thresholding operations to detect lanes lines of the road by creating a Binary image where the points are lane areas.

    To be able to detect the lanes, a slinding window search approach was implemented to search for the lane lines where the most binary activations are available in the image, and search around polynomial approach was implemented to search for lanes within a certain margin of a previously successful detection.

    As a final step, the detected lanes are drawn back on to the undistorted image, by applying an inverse persepective transform to get the unwarped image then drawing the lanes and their filling on the image.

    A configuration json file is provided where all the thresholding parameters are stored there, as well as the calibration images directory, test_images directory and the name of the video to be executed. This makes it easier for the user to just do the changes in the json file without the need to touch the code at all.

    ```json
    {
        "calibrationPath": "camera_cal",
        "chessboard": {
            "columns": "9",
            "rows": "6"
        },
        "colorThreshold": {
            "channel": "s_channel",
            "colorspace": "hls",
            "threshold": {
                "high": "255",
                "low": "90"
            }
        },
        "datapath": "test_images",
        "video": "harder_challenge_video.mp4",
        "videoMode": "True",
        "testImagesMode": "False",
        "saveMode": "False",
        "perspectiveTransform": {
            "src": {
                "ul_offset": {
                    "x": "-30",
                    "y": "80"
                },
                "ll_offset": {
                    "x": "-420",
                    "y": "0"
                },
                "lr_offset": {
                    "x": "550",
                    "y": "0"
                },
                "ur_offset": {
                    "x": "60",
                    "y": "80"
                }
            }
        },
        "threshold": {
            "dir_thresh": {
                "high": "1.3",
                "low": "0.7"
            },
            "grad_thresh": {
                "high": "100",
                "low": "20"
            },
            "kernel_size": "3",
            "mag_thresh": {
                "high": "170",
                "low": "30"
            }
        },
        "laneFinding": {
            "ym_per_pix": "30/720",
            "xm_per_pix": "3.7/700",
            "nwindows": "9",
            "margin": "100",
            "minpix": "50"
        }
    }
    ```

    To run the pipeline, this command need to be executed:

    ```bash
    python3 main.py config.json
    ```

2. Problems/Issues
   
   As shown from the results above, the pipeline which was developed performs really well on the test images and the project video with a fairly robust performance. This is due to the fact that the roads in basically ideal conditions, with fairly distinct lane lines, and on a clear day.
   
   The problems I encountered for the other videos were almost exclusively due to lighting conditions, shadows, discoloration, etc.
