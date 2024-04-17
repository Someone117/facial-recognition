import cv2
# import the necessary packages 
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import os
import math
from PIL import Image
import mtcnn
import matplotlib.pyplot as mpl


video_capture = cv2.VideoCapture(0)

from skimage import feature
import numpy as np
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist


def Distance(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx, ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.LANCZOS)
  return image

face_detector = mtcnn.MTCNN()

def detect_faces(vid):
    face_roi = face_detector.detect_faces(vid)
    cv2.imwrite("image.jpg", vid)

    image =  Image.open("image.jpg")

    if(len(face_roi) == 0):
        return None

    left_eye = face_roi[0]['keypoints']['left_eye']
    right_eye = face_roi[0]['keypoints']['right_eye']
    return CropFace(image, left_eye, right_eye, offset_pct=(0.27,0.27), dest_sz=(200,200)).save("image.jpg")

# tries to use LocalBinaryPatterns
from skimage import feature
import numpy as np
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
args = vars(ap.parse_args())
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)

# train a Linear SVM on the data
# model = LinearSVC(C=100.0, random_state=42)
# model.fit(data, labels)

from nn import NeuralNetwork as nn

labels2 = []
for i in range(len(labels)):
    if labels == "s":
        labels2.append(1)
    else:
        labels2.append(0)

network = nn(0.1, len(data[0]))
training_err = network.train(data, labels2, 1000)

mpl.plot(training_err)
mpl.xlabel("Iterations")
mpl.ylabel("Error for all training instances")
mpl.savefig("cumulative_error.png")

def detect_nn(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(vid)
    cv2.imwrite("image.jpg", vid)
    gray = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    arr = []
    for i in range(len(hist)):
      arr.append(hist[i])
    cv2.putText(vid, str(network.predict(arr)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    
    detect_nn(video_frame)

    # crop = detect_faces(video_frame)
    # if crop is None:
    #     print("no face")
    #     cv2.imshow(
    #     "My Face Detection Project", video_frame
    #     )  # display the processed frame in a window named "My Face Detection Project"
    #     break
    # gray = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2GRAY)
    # hist = desc.describe(gray)
    # prediction = model.predict(hist.reshape(1, -1))
	
	# display the image and the prediction
    # cv2.putText(video_frame, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		# 1.0, (0, 0, 255), 3)

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

cv2.destroyAllWindows()