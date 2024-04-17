# tries to recognize a face from a webcam
from sklearn.svm import LinearSVC, LinearSVR
import argparse
import os
import math
from PIL import Image
import mtcnn
import matplotlib.pyplot as mpl
import cv2
from imutils import paths
from nn import NeuralNetwork as nn


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def DistanceHist(hist1, hist2):
	hist3 = hist1
	for i in range(len(hist1)):
		# find the distance of each part of hist1 and hist2 and save to hist3
		hist3[i] = abs(hist1[i] - hist2[i])
	return hist3

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

# find the min and max distance of each part of the hist with all similar and with all different to generate a threshold for each part
# create lower and upper bound histograms

# x = hist.size
# y = hist value
# z = label


# totalXS = []
# totalYS = []
# totalXB = []
# totalYB = []

# for i in range(len(data)):
# 	if labels[i] == "s":
# 		totalXS.append(range(len(hist)))
# 		totalYS.append(data[i])
# 	else:
# 		totalXB.append(range(len(hist)))
# 		totalYB.append(data[i])

# mpl.scatter(totalXS, totalYS, c='b')
# mpl.scatter(totalXB, totalYB, c='r')
# mpl.show()


# lowerBoundSame = [1000.0] * len(hist)
# upperBoundSame = [0.0] * len(hist)
# lowerBoundDiff = [1000.0] * len(hist)
# upperBoundDiff = [0.0] * len(hist)
# averageS = [0.0] * len(hist)
# dataS = 0
# averageB = [0.0] * len(hist)
# dataB = 0

# for i in range(len(data)):
# 	if labels[i] == "s":
# 		for j in range(len(hist)):
# 			averageS[j] += data[i][j]
# 		dataS += 1
# 	else:
# 		for j in range(len(hist)):
# 			averageB[j] += data[i][j]
# 		dataB += 1

# for i in range(len(averageS)):
# 	averageS[i] /= dataS
# 	averageB[i] /= dataB

# # generate threshold for each part of the histogram
# for i in range(len(data)):
# 	if(labels[i] != "s"):
# 		for j in range(len(averageS)):
# 			upperBoundDiff[j] = max(data[i][j], upperBoundDiff[j])
# 			lowerBoundDiff[j] = min(data[i][j], lowerBoundDiff[j])
# 	else:
# 		for j in range(len(averageS)):
# 			upperBoundSame[j] = max(data[i][j], upperBoundSame[j])
# 			lowerBoundSame[j] = min(data[i][j], lowerBoundSame[j])

# newUpperSame = []
# newLowerSame = []

# for i in range(len(averageS)):
# 	# use the quartiles instead of ranges
# 	newUpperSame.append(np.percentile([upperBoundSame[i], averageS[i], lowerBoundSame[i]], 50))
# 	newLowerSame.append(np.percentile([upperBoundSame[i], averageS[i], lowerBoundSame[i]], 50))

video_capture = cv2.VideoCapture(0)

labels2 = []
for i in range(len(labels)):
    if labels == "s":
        labels2.append(1)
    else:
        labels2.append(0)

network = nn(0.1, len(hist))
training_err = network.train(data, labels2, 1000)

mpl.plot(training_err)
mpl.xlabel("Iterations")
mpl.ylabel("Error for all training instances")
mpl.savefig("cumulative_error.png")

# def detect_bounding_box(vid):
#     gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
#     for (x, y, w, h) in faces:
#         cropped_face = gray_image[y:y+h, x:x+w]
#         # resize image to 200 by 200
#         resized_face = cv2.resize(cropped_face, (200, 200)) # use for ai
#         hist = desc.describe(resized_face)
#         for i in range(len(hist)):
#             if hist[i] > newUpperSame[i]:
#                 hist[i] = hist[i] - newUpperSame[i]
#             elif hist[i] < newLowerSame[i]:
#                 hist[i] = newLowerSame[i] - hist[i]
#             else:
#                 hist[i] = 0       
#         # average the hist
#         total = 0.0
#         for i in range(len(hist)):
#             total += hist[i]
#         total/=len(hist)
#         if(total < 0.004):
#             cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(vid, str(total), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def detect_nn(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cropped_face = gray_image[y:y+h, x:x+w]
        # resize image to 200 by 200
        resized_face = cv2.resize(cropped_face, (200, 200)) # use for ai
        hist = desc.describe(resized_face)
        arr = []
        for i in range(len(hist)):
            arr.append(hist[i])
        cv2.putText(vid, str(network.predict(arr)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully
    
    detect_nn(
        video_frame
    )  # apply the function we created to the video frame
    cv2.imshow("Video", video_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()