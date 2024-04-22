import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os
import threading
import shutil

# Some useful resources: 
# - https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
# - http://ksimek.github.io/2012/08/22/extrinsic/
# - https://learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/

def capture_streams():

	class WebcamFrameCapturer():
		def __init__(self, previewName, camID, writePath):
			threading.Thread.__init__(self)
			self.previewName = previewName
			self.camID = camID
			self.saved_img_count = 0
			self.write_path = writePath

		def camPreview(self):
			print(f"Starting {self.previewName}")

			while rval:
				cv.imshow(self.previewName, frame)
				rval, frame = cam.read()
				# Do not block, just check for the key
				key = cv.waitKey(-1)
				if key == 32: # exit on space key
					# TODO: write image
					filename = self.previewName+"_"+str(self.saved_img_count)+".png"
					full_filename_path = os.path.join(self.write_path,self.previewName,filename)
					cv.imwrite(full_filename_path, frame) 
					print(full_filename_path)

					self.saved_img_count = self.saved_img_count + 1
					break
			cv.destroyWindow(self.previewName)


	# Update the ID's below based on which webcam corresponds to which device in /dev/videoX with X=id
	images_folder = '~/calib_imgs'


	# Setup left cam
	left_cam_name = "left"
	left_cam_id = 4
	left_cam = cv.VideoCapture(left_cam_id)
	left_cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
	left_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


	# Setup right cam
	right_cam_name = "right"
	right_cam_id = 2
	right_cam = cv.VideoCapture(right_cam_id)
	right_cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
	right_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


	# Delete old directory with images
	if os.path.exists(images_folder):
		shutil.rmtree(images_folder)

	os.makedirs(os.path.join(images_folder, "left"))
	os.makedirs(os.path.join(images_folder, "right"))

	# Get info
	if left_cam.isOpened():
		rval_left, frame_left = left_cam.read()
		height, width, channels = frame_left.shape
		print("Width: ", width)
		print("Height: ", height)
		print("Channels: ", channels)
	else:
		print("Failed to open left cam")
		return

	if right_cam.isOpened():
		rval_right, frame_right = right_cam.read()
		height, width, channels = frame_right.shape
		print("Width: ", width)
		print("Height: ", height)
		print("Channels: ", channels)
	else:
		print("Failed to open right cam")
		return

	saved_img_count = 0
	while 1:
		if left_cam.isOpened():
			rval_left, frame_left = left_cam.read()
		if right_cam.isOpened():
			rval_right, frame_right = right_cam.read()

		if(rval_right and rval_left):
			# Resize right to size of left for display purposes
			l_height, l_width, c = frame_left.shape
			r_height, r_width, c = frame_right.shape
			new_r_width = int(r_width * (l_height / r_height)) # preserve aspect ratio
			right_resized = cv.resize(frame_right, (new_r_width, l_height), interpolation = cv.INTER_LINEAR) # Resize to l/w of left
			concat_img = cv.hconcat([frame_left, right_resized])
			# Display
			cv.imshow("L/R pair", concat_img)

			key = cv.waitKey(1)
			if key == 32: # exit on space key
				# Write left image
				filename = left_cam_name+"_"+str(saved_img_count)+".png"
				full_filename_path = os.path.join(images_folder,left_cam_name,filename)
				cv.imwrite(full_filename_path, frame_left) 
				print(f"Wrote to {full_filename_path}")

				# Write right image
				filename = right_cam_name+"_"+str(saved_img_count)+".png"
				full_filename_path = os.path.join(images_folder,right_cam_name,filename)
				cv.imwrite(full_filename_path, frame_right) 
				print(f"Wrote to {full_filename_path}")

				saved_img_count = saved_img_count + 1


		# Exit condition
		if saved_img_count == 10:
			print("Enough images collected, image collection complete.")
			cv.destroyWindow(right_cam_name)
			cv.destroyWindow(left_cam_name)
			break	

def camera_calibrate(cam_name):
	# Get images from image folder
	images_folder = os.path.join('~/calib_imgs', cam_name, '*')
	images_names = sorted(glob.glob(images_folder))
	images = []
	for imname in images_names:
		im = cv.imread(imname, 1)
		images.append(im)

	# Checkerboard pattern - change if can't find the checkerboard
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	rows = 9 # num checkerboard rows
	columns = 6 # num checkerboard columns
	world_scaling = 1. # change this to the real world square size

	# Coordinates of squares in the checkerboard world space
	# This enumerates the entire grid from (0,0,0)...(n,n,0)
	objp = np.zeros((rows*columns, 3), np.float32)
	objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
	objp = world_scaling * objp

	# Frame dimensions. Frames should be the same size.
	width = images[0].shape[1]
	height = images[0].shape[0]

	# Pixel coordinates of checkerboards
	imgpoints = [] # 2d points in image plane.

	# Coordinates of the checkerboard in checkerboard world space
	objpoints = [] # 3d point in real world space

	for frame in images:
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Find the checkerboard (no need for the corners)
		ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

		if ret == True:
			# Convolution size used to improve corner detection. Don't make too large
			conv_size = (11, 11)

			# OpenCV can attempt to improve the checkerboard coordinates
			corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)

			# Visualize corners
			cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
			cv.imshow(cam_name, frame)
			key = cv.waitKey(500)

			objpoints.append(objp)
			imgpoints.append(corners)

	cv.destroyWindow(cam_name)
	ret, K, distortion_coeffs, R, T = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
	print(f"RMSE = {ret}")
	return K, distortion_coeffs

def stereo_calibrate_and_rectify(K_left, dist_left, K_right, dist_right):
	images_folder = '~/calib_imgs'
	left_images_folder = os.path.join(images_folder, "left", '*')
	left_images_names = sorted(glob.glob(left_images_folder))
	right_images_folder = os.path.join(images_folder, "right", '*')
	right_images_names = sorted(glob.glob(right_images_folder))

	left_images = []
	right_images = []
	for left, right in zip(left_images_names, right_images_names):
		_im = cv.imread(left, 1)
		left_images.append(_im)

		_im = cv.imread(right, 1)
		right_images.append(_im)
 

	#change this if stereo calibration not good.
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
	 
	rows = 9 #number of checkerboard rows.
	columns = 6 #number of checkerboard columns.
	world_scaling = 1. #change this to the real world square size. Or not.
	 
	#coordinates of squares in the checkerboard world space
	objp = np.zeros((rows*columns,3), np.float32)
	objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
	objp = world_scaling* objp
	 
	#frame dimensions. Frames should be the same size.
	width = left_images[0].shape[1]
	height = left_images[0].shape[0]
	 
	#Pixel coordinates of checkerboards
	imgpoints_left = [] # 2d points in image plane.
	imgpoints_right = []
	 
	#coordinates of the checkerboard in checkerboard world space.
	objpoints = [] # 3d point in real world space
	 
	for frame_left, frame_right in zip(left_images, right_images):
	    gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
	    gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)
	    c_ret_left, corners_left = cv.findChessboardCorners(gray_left, (rows, columns), None)
	    c_ret_right, corners_right = cv.findChessboardCorners(gray_right, (rows, columns), None)
	 
	    if c_ret_left == True and c_ret_right == True:
	        corners_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
	        corners_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
	 
	        cv.drawChessboardCorners(frame_left, (rows,columns), corners_left, c_ret_left)
	        cv.drawChessboardCorners(frame_right, (rows,columns), corners_right, c_ret_right)

	        concat = np.hstack((frame_left, frame_right))
	        cv.imshow('left, right pair', concat)
	        key = cv.waitKey(500)

	        objpoints.append(objp)
	        imgpoints_left.append(corners_left)
	        imgpoints_right.append(corners_right)

	cv.destroyWindow("left, right pair")

	# Skip individual camera calibration via fix intrinsic flag
	stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
	ret, CM_L, dist_L, CM_R, dist_R, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, K_left, distortion_coeffs_left,
K_right, distortion_coeffs_right, (width, height), criteria = criteria, flags = stereocalibration_flags)
	print(f"Stereo RMSE = {ret}")

	# Rectify the cameras
	# rect_l brings points in the left cameras's coordinate system to points in the rectified left camera's coordinate system.
	# rect_r brings points in the right cameras's coordinate system to points in the rectified right camera's coordinate system.
	# proj_mat_l projects points given in the rectified left camera coordinate system into the rectified left camera's image
	# proj_mat_r does the same as above but for right cam
	# roi l/r are the output rectangles inside the rectified images where all the pixels are valid 
	# Q is disparity to depth mapping matrix (see reprojectImageTo3D function)
	rectify_scale=1
	rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR= cv.stereoRectify(CM_L, dist_L, CM_R, dist_R, (width, height), R, T, rectify_scale,(640,480))

	# Save undistortion map
	Left_Stereo_Map= cv.initUndistortRectifyMap(CM_L, dist_L, rect_l, proj_mat_l,
                                             (640, 480), cv.CV_16SC2)
	Right_Stereo_Map= cv.initUndistortRectifyMap(CM_R, dist_R, rect_r, proj_mat_r,
                                              (640, 480), cv.CV_16SC2)

	# Verify rectification
	imgL = left_images[0]
	imgR = right_images[0]
	# cv.imshow("Left image before rectification", imgL)
	# cv.imshow("Right image before rectification", imgR)

	img_left_rectified = cv.remap(imgL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
	img_right_rectified = cv.remap(imgR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

	# Draw the bounding box on the left rectified image
	xL, yL, wL, hL = roiL
	cv.rectangle(img_left_rectified, (xL, yL), (xL+wL, yL+hL), (0, 255, 0), 2)  # Green rectangle with a thickness of 2

	# Draw the bounding box on the right rectified image
	xR, yR, wR, hR = roiR
	cv.rectangle(img_right_rectified, (xR, yR), (xR+wR, yR+hR), (0, 0, 255), 2)  # Red rectangle with a thickness of 2

	# Draw 10 horizontal lines across the concatenated image
	combined_image = np.hstack((img_left_rectified, img_right_rectified))
	num_lines = 10
	step = 720 // num_lines
	for i in range(num_lines):
	    y = step * i
	    cv.line(combined_image, (0, y), (combined_image.shape[1], y), (255, 255, 0), 1)  # Yellow lines

	cv.imshow("Rectified Images with Lines", combined_image)
	# cv.imshow("Left image after rectification", img_left_rectified)
	# cv.imshow("Right image after rectification", img_right_rectified)


	cv.waitKey(0)
	cv.destroyWindow("Rectified Images with Lines")

	return R, T, Left_Stereo_Map, Right_Stereo_Map, roiL, roiR

def stereo_match(R, T, Left_Stereo_Map, Right_Stereo_Map, roiL, roiR):

	# Create a StereoSGBM object
	min_disp = 0  # Minimum possible disparity value
	num_disp = 16*6  # Number of disparities to consider. Should be divisible by 16.
	block_size = 3  # Size of the block window. Must be odd.

	stereo = cv.StereoSGBM_create(
	    minDisparity=min_disp,
	    numDisparities=num_disp,
	    blockSize=block_size,
	    P1=8 * 3 * block_size**2,  # 8*number of image channels*blockSize^2
	    P2=32 * 3 * block_size**2, # 32*number of image channels*blockSize^2
	    disp12MaxDiff=1,
	    uniquenessRatio=15,
	    speckleWindowSize=100,
	    speckleRange=32,
	    preFilterCap=63,
	    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
	)

	# Get images
	images_folder = '~/calib_imgs'
	left_images_folder = os.path.join(images_folder, "left", '*')
	left_images_names = sorted(glob.glob(left_images_folder))
	right_images_folder = os.path.join(images_folder, "right", '*')
	right_images_names = sorted(glob.glob(right_images_folder))

	left_images = []
	right_images = []
	for left, right in zip(left_images_names, right_images_names):
		_im = cv.imread(left, 1)
		left_images.append(_im)

		_im = cv.imread(right, 1)
		right_images.append(_im)

	for frame_left, frame_right in zip(left_images, right_images):
		# TODO: Stream left and right cameras
		img_left_rectified = cv.remap(frame_left,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
		img_right_rectified = cv.remap(frame_right,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
		# Draw the bounding box on the left rectified image
		xL, yL, wL, hL = roiL
		cv.rectangle(img_left_rectified, (xL, yL), (xL+wL, yL+hL), (0, 255, 0), 2)  # Green rectangle with a thickness of 2

		# Draw the bounding box on the right rectified image
		xR, yR, wR, hR = roiR
		cv.rectangle(img_right_rectified, (xR, yR), (xR+wR, yR+hR), (0, 0, 255), 2)  # Red rectangle with a thickness of 2

		# Compute disparity, normalize, apply colormap, and display
		disparity = stereo.compute(img_left_rectified, img_right_rectified).astype(np.float32) / 16.0
		disp_norm = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		disp_color = cv.applyColorMap(disp_norm, cv.COLORMAP_JET)

		# Draw valid ROI over depth map
		depth_roi = cv.getValidDisparityROI(roiL, roiR, min_disp, num_disp, block_size);
		xL, yL, wL, hL = depth_roi
		cv.rectangle(disp_color, (xL, yL), (xL+wL, yL+hL), (0, 255, 0), 2)  # Green rectangle with a thickness of 2

		# Show depth and rectified image pair
		combined_image_rect = np.hstack((img_left_rectified, img_right_rectified, disp_color))
		cv.imshow("Rect Img + Colored Disparity Map", combined_image_rect)
		k = cv.waitKey(0)

	cv.destroyWindow("Rect Img + Colored Disparity Map")

	# Setup left cam
	left_cam_name = "left"
	left_cam_id = 4
	left_cam = cv.VideoCapture(left_cam_id)
	left_cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
	left_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


	# Setup right cam
	right_cam_name = "right"
	right_cam_id = 2
	right_cam = cv.VideoCapture(right_cam_id)
	right_cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
	right_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

	# Get info
	if left_cam.isOpened():
		rval_left, frame_left = left_cam.read()

	else:
		print("Failed to open left cam")
		return

	if right_cam.isOpened():
		rval_right, frame_right = right_cam.read()
		height, width, channels = frame_right.shape
	else:
		print("Failed to open right cam")
		return

	while 1:
		if left_cam.isOpened():
			rval_left, frame_left = left_cam.read()
		if right_cam.isOpened():
			rval_right, frame_right = right_cam.read()

		img_left_rectified = cv.remap(frame_left,Left_Stereo_Map[0],Left_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
		img_right_rectified = cv.remap(frame_right,Right_Stereo_Map[0],Right_Stereo_Map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
		# Draw the bounding box on the left rectified image
		xL, yL, wL, hL = roiL
		cv.rectangle(img_left_rectified, (xL, yL), (xL+wL, yL+hL), (0, 255, 0), 2)  # Green rectangle with a thickness of 2

		# Draw the bounding box on the right rectified image
		xR, yR, wR, hR = roiR
		cv.rectangle(img_right_rectified, (xR, yR), (xR+wR, yR+hR), (0, 0, 255), 2)  # Red rectangle with a thickness of 2

		# Compute disparity, normalize, apply colormap, and display
		disparity = stereo.compute(img_left_rectified, img_right_rectified).astype(np.float32) / 16.0
		disp_norm = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		disp_color = cv.applyColorMap(disp_norm, cv.COLORMAP_JET)

		# Draw valid ROI over depth map
		depth_roi = cv.getValidDisparityROI(roiL, roiR, min_disp, num_disp, block_size);
		xL, yL, wL, hL = depth_roi
		cv.rectangle(disp_color, (xL, yL), (xL+wL, yL+hL), (0, 255, 0), 2)  # Green rectangle with a thickness of 2

		# Show depth and rectified image pair
		combined_image_rect = np.hstack((img_left_rectified, img_right_rectified, disp_color))
		cv.imshow("Rect Img + Colored Disparity Map", combined_image_rect)
		key = cv.waitKey(1)
		if key == 32: # exit on space key
			break



if __name__=='__main__':
	# Uncomment this line to do a new chessboard capture. This is only necessary if you've moved a camera after a good calibration, or if your calibration was bad and you need to re-do it.
	# capture_streams()

	# Calibrate the cameras
	K_left, distortion_coeffs_left = camera_calibrate("left")
	K_right, distortion_coeffs_right = camera_calibrate("right")
	R, T, Left_Stereo_Map, Right_Stereo_Map, roiL, roiR = stereo_calibrate_and_rectify(K_left, distortion_coeffs_left, K_right, distortion_coeffs_right)

	# Run stereo matching
	stereo_match(R, T, Left_Stereo_Map, Right_Stereo_Map, roiL, roiR)

