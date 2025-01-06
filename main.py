import cv2
import numpy as np

print("Installed opencv version", cv2.__version__)
print("Installed numpy version", np.__version__)

FEATURES = 100

# params for ShiTomasi corner detection
feature_params = dict(
	maxCorners=FEATURES,
	qualityLevel=0.3,
	minDistance=20,
	blockSize=7
)

# Parameters for lucas kanade optical flow
lk_params = dict(
	winSize=(30, 30),
	maxLevel=0,
	criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

vidPath = "data/level_1.mp4"
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
while 1:
	ret, frame = cap.read()
	if frame is None:
		break
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st == 1]
	good_old = p0[st == 1]

	# draw the tracks
	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.astype('int').ravel()
		c, d = old.astype('int').ravel()
		mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
		frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
	img = cv2.add(frame, mask)

	# Write the frame to the output video
	out.write(img)

	cv2.imshow('frame', img)

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
out.release()
