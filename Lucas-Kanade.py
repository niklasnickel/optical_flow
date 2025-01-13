#################################
### Lucas-Kanade Optical Flow ###
#################################

# Import the required libraries
import cv2
import numpy as np

print("Installed opencv version", cv2.__version__)
print("Installed numpy version", np.__version__)

#####################
# Parameters
#####################

FEATURES = 100

# Parameters for feature selection
# See: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga4055896d9ef77dd3cacf2c5f60e13f1c
feature_params = dict(
	maxCorners=FEATURES,  # Maximum number of features to track
	qualityLevel=0.3,  # Minimal eigenvalue of the feature matrix
	minDistance=20,  # Minimum distance between features
	blockSize=7  # Dimension of the search matrix W (used for feature selection)
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
	winSize=(30, 30),  # Dimension of the search matrix W (used for optical flow detection)
	maxLevel=3,  # Maximum pyramid level
	criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Convergence criteria
)

#####################
# Preparation
#####################

vidPath = "data/level_2.mp4"
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

#####################
# Tracking
#####################

while 1:
	ret, frame = cap.read()
	if frame is None:
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	# Calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st == 1]
	good_old = p0[st == 1]

	# Draw the tracks
	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.astype('int').ravel()
		c, d = old.astype('int').ravel()
		mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
		frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
	img = cv2.add(frame, mask)

	# Write the frame to the output video
	out.write(img)

	cv2.imshow('Lucas-Kanade Optical Flow', img)

	if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
		break  # This line is necessary to display the frame during calculation

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1, 1, 2)

# Close windows and release video capture and writer
cv2.destroyAllWindows()
cap.release()
out.release()
