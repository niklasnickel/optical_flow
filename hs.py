import time

import cv2
import numpy as np
from IPython.core.display import display, HTML

print("Installed opencv version", cv2.__version__)
print("Installed numpy version", np.__version__)

display(HTML("<style>"
			 + "#notebook { padding-top:0px !important; } "
			 + ".container { width:100% !important; } "
			 + ".end_space { min-height:0px !important; } "
			 + "</style>"))

FEATURES = 100

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=FEATURES,
					  qualityLevel=0.3,
					  minDistance=7,
					  blockSize=7)

# Parameters for Farneback optical flow
hs_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

vidPath = "data/bamboo_1/clean.mp4"
output_video_file = "output_video.mp4"

cap = cv2.VideoCapture(vidPath)

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, 30.0, (frame_width, frame_height))

# Create some random colors
color = np.random.randint(0, 255, (FEATURES, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
etime = 0
while (1):
	ret, frame = cap.read()
	if frame is None:
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	# calculate optical flow
	t = time.perf_counter()
	flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **hs_params)
	etime += (time.perf_counter() - t)

	# Compute the magnitude and angle of the flow
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Create an HSV image
	hsv = np.zeros_like(old_frame)
	hsv[..., 1] = 255

	# Set the hue and value according to the flow
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

	# Convert HSV to BGR
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# Write the frame to the output video
	out.write(bgr)

	cv2.imshow('frame', bgr)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

	# Now update the previous frame
	old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
out.release()
