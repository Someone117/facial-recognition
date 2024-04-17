import keras
import tensorflow as tf
import numpy as np
import keras_vggface
from keras_vggface.vggface import VGGFace
import mtcnn
import matplotlib.pyplot as mpl
import keras_vggface.utils
import PIL
import os
import os.path

print('a')
photo =  mpl.imread('training/b/WIN_20240403_12_57_00_Pro.jpg')
print('b')
face_detector = mtcnn.MTCNN()
print('starting')
face_roi = face_detector.detect_faces(photo)

print(face_roi) # detect a face


x1, y1, width, height = face_roi[0]['box']
x2, y2 = x1 + width, y1 + height
face = photo[y1:y2, x1:x2]
print(face.shape)