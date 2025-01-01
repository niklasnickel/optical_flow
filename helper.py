import cv2
import os

# Directory containing the image sequence
image_folder = 'data/bamboo_1'
# Output video file
video_file = 'data/bamboo_1.mp4'

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