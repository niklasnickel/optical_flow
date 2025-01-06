import os

import cv2
import numpy as np


def convert_img_sequence_to_video(image_folder, video_file):
	# Get list of images in the directory
	images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
	images.sort()  # Ensure the images are in the correct order

	# Read the first image to get the dimensions
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))

	for image in images:
		img_path = os.path.join(image_folder, image)
		frame = cv2.imread(img_path)
		video.write(frame)

	# Release the video writer
	video.release()
	cv2.destroyAllWindows()


def read_flow_file(file_path):
	with open(file_path, 'rb') as f:
		magic = np.fromfile(f, np.float32, count=1)
		if magic != 202021.25:
			raise ValueError('Magic number incorrect. Invalid .flo file')
		w = np.fromfile(f, np.int32, count=1)[0]
		h = np.fromfile(f, np.int32, count=1)[0]
		data = np.fromfile(f, np.float32, count=2 * w * h)
		return np.resize(data, (h, w, 2))


def convert_flow_sequence_to_video(flow_folder, video_file):
	# Get list of .flo files in the directory
	flow_files = [f for f in os.listdir(flow_folder) if f.endswith(".flo")]
	flow_files.sort()  # Ensure the files are in the correct order

	# Read the first flow file to get the dimensions
	flow = read_flow_file(os.path.join(flow_folder, flow_files[0]))
	height, width, _ = flow.shape

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))

	for flow_file in flow_files:
		flow_path = os.path.join(flow_folder, flow_file)
		flow = read_flow_file(flow_path)

		# Compute the magnitude and angle of the flow
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

		# Create an HSV image
		hsv = np.zeros((height, width, 3), dtype=np.uint8)
		hsv[..., 1] = 255

		# Set the hue and value according to the flow
		hsv[..., 0] = ang * 180 / np.pi / 2
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

		# Convert HSV to BGR
		bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		# Write the frame to the output video
		video.write(bgr)

	# Release the video writer
	video.release()
	cv2.destroyAllWindows()


# Directory containing the image sequence
image_folder = 'MPI-Sintel-complete/training/clean/bamboo_2'
flow_folder = 'MPI-Sintel-complete/training/flow/bamboo_2'

# Output video file
video_file = 'testing_data/bamboo_2/flow.mp4'

# convert_img_sequence_to_video(image_folder, video_file)
convert_flow_sequence_to_video(flow_folder, video_file)