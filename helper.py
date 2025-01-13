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
		hsv[..., 2] = 20 * mag
		# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

		# Convert HSV to BGR
		bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		# Write the frame to the output video
		video.write(bgr)

	# Release the video writer
	video.release()
	cv2.destroyAllWindows()


def count_frames(video_path):
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error: Could not open video {video_path}")
		return 0

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()
	return frame_count


def compute_video_differences(video_path1, video_path2, output_video_path):
	cap1 = cv2.VideoCapture(video_path1)
	cap2 = cv2.VideoCapture(video_path2)

	if not cap1.isOpened() or not cap2.isOpened():
		print("Error: Could not open one of the video files.")
		return

	# Get the width and height of the frames
	frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

	average_epe = []

	while True:
		ret1, frame1 = cap1.read()
		ret2, frame2 = cap2.read()

		if not ret1 or not ret2:
			break

		frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
		frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

		frame_1_x = frame1_hsv[:, :, 2] * np.cos(frame1_hsv[:, :, 0] / 255)
		frame_1_y = frame1_hsv[:, :, 2] * np.sin(frame1_hsv[:, :, 0] / 255)

		frame_2_x = frame2_hsv[:, :, 2] * np.cos(frame2_hsv[:, :, 0] / 255)
		frame_2_y = frame2_hsv[:, :, 2] * np.sin(frame2_hsv[:, :, 0] / 255)

		epe = np.sqrt((frame_1_x - frame_2_x) ** 2 + (frame_1_y - frame_2_y) ** 2)

		# Compute the absolute difference between the frames
		diff = cv2.absdiff(frame1, frame2)

		# Write the frame to the output video
		out.write(diff)

		# Display the resulting frame
		cv2.imshow('Difference', diff)

		average_epe.append(np.mean(epe) / 20)

	cap1.release()
	cap2.release()
	out.release()
	cv2.destroyAllWindows()

	return np.mean(average_epe)


# Directory containing the image sequence
image_folder = 'MPI-Sintel-complete/training/clean/bamboo_2'
flow_folder = 'MPI-Sintel-complete/training/flow/bamboo_1'

# Output video file
video_file = 'testing_data/bamboo_1/flow.mp4'

# convert_img_sequence_to_video(image_folder, video_file)
# convert_flow_sequence_to_video(flow_folder, video_file)

test_video_path = "output_video.mp4"
validation_video_path = "data/level_3-solution.mp4"

error = compute_video_differences(test_video_path, validation_video_path, "output_video_diff.mp4")
print(error)
