#################################
### Horn-Schunck Optical Flow ###
#################################

# Import the required libraries
import cv2
import numpy as np

from helper import compute_video_differences

print("Installed opencv version", cv2.__version__)
print("Installed numpy version", np.__version__)

#####################
# Parameters
#####################

# Parameters for the Zach-Pock implementation of the Horn-Schunck optical flow
# See: https://docs.opencv.org/3.3.0/dc/d47/classcv_1_1DualTVL1OpticalFlow.html and https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22
params = dict(
	# tau=0.125, # Time step of the numerical scheme
	lambda_=0.10,  # Smoothness parameter (lower numbers are smoother)
	nscales=5,  # Number of scales in the image pyramid
	warps=5,  # Number of warps per scale .Represents the number of times that I1(x+u0) and grad(I1(x+u0)) are computed per scale.
	epsilon=0.01,  # Stopping criterion threshold used in the numerical scheme
	medianFiltering=5,  # Size of the filter to be applied to the flow field at each pyramid level. Use 1 to disable filtering.
	useInitialFlow=False,  # Use the input flow as an initial flow approximation. (False since we don't have an initial flow)

	innnerIterations=30,  # Don't bother. Ask me if you are interested. ;)
	outerIterations=10,  # Don't bother. Ask me if you are interested. ;)
	theta=0.3  # Don't bother. Ask me if you are interested. ;)
)

#####################
# Preparation
#####################

vidPath = "testing_data/bamboo_1/clean.mp4"
output_video_file = "output_video.mp4"
validation_video_path = "data/level_3-solution.mp4"

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
frame_number = 0

#####################
# Tracking
#####################

while 1:
	ret, frame = cap.read()
	if frame is None:
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	print(f"Processing frame {frame_number}")

	# calculate optical flow
	optical_flow = cv2.optflow.createOptFlow_DualTVL1()
	# optical_flow.setTau(params['tau'])
	optical_flow.setLambda(params['lambda_'])
	optical_flow.setTheta(params['theta'])
	optical_flow.setScalesNumber(params['nscales'])
	optical_flow.setWarpingsNumber(params['warps'])
	optical_flow.setEpsilon(params['epsilon'])
	optical_flow.setInnerIterations(params['innnerIterations'])
	optical_flow.setOuterIterations(params['outerIterations'])
	optical_flow.setMedianFiltering(params['medianFiltering'])
	optical_flow.setUseInitialFlow(params['useInitialFlow'])
	flow = optical_flow.calc(old_gray, frame_gray, None)

	# Compute the magnitude and angle of the flow
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

	# Create an HSV image
	hsv = np.zeros_like(old_frame)
	hsv[..., 1] = 255

	# Set the hue and value according to the flow
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 2] = 20 * mag

	# Convert HSV to BGR
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# Write the frame to the output video
	out.write(bgr)

	cv2.imshow('Horn-Schunck Optical Flow', bgr)

	if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
		break  # This line is necessary to display the frame during calculation

	# Now update the previous frame
	old_gray = frame_gray.copy()

	frame_number += 1

cv2.destroyAllWindows()
cap.release()
out.release()

print(f"Your video is ready! Check the file {output_video_file}.")
print("Calculating the endpoint error (non normalized)...")
# Compute differences between the output video and the validation video
endpoint_error = compute_video_differences(output_video_file, validation_video_path, "output_video_diff.mp4")
print(f"Your endpoint error was: {endpoint_error}.")
print(f"The error for the Horn-Schunck optical flow for a comparable scene is 1.33 (average: 8.739).") # http://sintel.is.tue.mpg.de/hero?flow_type=Error&method_id=51&metric_id=0&selected_pass=1
print(f"The current world record is 0.375 (average: 0.787) for the whm_flow_sintel_1222 algorithm.") # http://sintel.is.tue.mpg.de/hero?flow_type=Error&method_id=4384&metric_id=0&selected_pass=1