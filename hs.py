import cv2
import numpy as np

print("Installed opencv version", cv2.__version__)
print("Installed numpy version", np.__version__)

### Horn-Schunck Optical Flow ###

# Parameters for Farneback optical flow
hs_params = dict(
	pyr_scale=0.5,
	levels=4,
	winsize=15,
	iterations=6,
	poly_n=5,
	poly_sigma=1.2,
	flags=0
)

vidPath = "testing_data/bamboo_1/clean.mp4"
output_video_file = "output_video.mp4"

cap = cv2.VideoCapture(vidPath)

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, 30.0, (frame_width, frame_height))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
etime = 0
while 1:
	ret, frame = cap.read()
	if frame is None:
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	# calculate optical flow
	flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **hs_params)

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

	# Now update the previous frame
	old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
out.release()
