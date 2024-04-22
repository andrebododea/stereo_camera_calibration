# Motivation
Having some fun calibrating two random webcams and creating depth maps. 

<img src="https://github.com/andrebododea/stereo_camera_calibration/assets/9446419/570ce853-6188-40f9-965a-51f75c86b963" width="450">

# Instructions

## Recording initial calibration data
When you initially run calibration, you'll need to capture some chessboard images (default is currently set to 10). To run this process, simply run the script with the requisite flag: `python3 stereo_calibration.py --chessboard-capture`

At the end of the process you will have synced left/right images in separate directories ready to use for calibration

<img src="https://github.com/andrebododea/stereo_camera_calibration/assets/9446419/d73dff32-0f02-40ac-ba8f-b2050f893735" width="450">


Collect images of the chessboard at different positions and orientations within both camera views, trying to cover the entire field of view of both cameras as much as possible. When you have collected enough saples, the stream will close and the commandline will alert you that this portion of the process is finished. Once that's complete, verify in the terminal that your RMSE for each of the calibration steps is below 0.3, otherwise it's recommended that you re-run the chessboard capture. 

You can print out the standard 9x6 OpenCV chessboard calibration pattern from [here](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png). If you use a calibration pattern with different characteristics, you will need to update the chessboard parameters in the code.


## Running calibration and running stereo
If you do not specify the above flag, the program will just operate on the saved chessboard images. It will calibrate each camera individually, then do stereo rectification for the camera pair. At the end of the stereo rectification process, it'll show horizontal lines spanning across one of the undistorted and stereo rectified image pairs. Use these lines to verify that the two images are rectified correctly (e.g. visually confirm that a given point intersecting with the horizontal line in the left image also intersects with the same point in the right image).
![image](https://github.com/andrebododea/stereo_camera_calibration/assets/9446419/f5b32301-92de-4b22-b560-968d0e7fb1ef)


At the end it'll stream a depth map using the undistorted and rectified images, and capture the valid region of interest in the resultant depth map ([quick video here](https://github.com/andrebododea/stereo_camera_calibration/assets/9446419/e7cbbb12-9d7d-47e4-9dbf-ba0e1e8278e3)).


Depth map isn't great, so future steps to improve the current stereo matching approach could involve tuning the parameters of the SGBM algorithm and adding some filtering (temporal, hole filling, etc). Significantly more accurate stereo depth maps (that don't rely on any sort of structured light emitter) would likely need to be achieved by a stereo depth DNN.
