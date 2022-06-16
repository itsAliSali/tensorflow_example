# dataset: https://data.caltech.edu/records/20086

import scipy.io
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from visualize import plot


data_dir = ['Faces', 'Motorbikes', 'airplanes']
label_dir = ['Faces_2', 'Motorbikes_16', 'Airplanes_Side_2']

lbl_map = {'Faces': 0, 'Motorbikes': 1, 'airplanes': 2}

train_data = []
train_lbl = []

for d_dirname, lbl_dirname in zip(data_dir, label_dir):
    d_dirpath = f'./caltech-101/101_ObjectCategories/{d_dirname}/'
    data_fnames = sorted(os.listdir(d_dirpath))
    for d_fname in data_fnames:
        d_path = os.path.join(d_dirpath, d_fname)
        img = cv2.imread(d_path)
        img = cv2.resize(img, (100, 75))
        train_data.append(img)
        # cv2.imshow('a', img)
        # cv2.waitKey(0)

    lbl_dirpath = f'./caltech-101/Annotations/{lbl_dirname}/'
    lbl_fnames = sorted(os.listdir(lbl_dirpath))
    for lbl_fname in lbl_fnames:
        lbl_path = os.path.join(lbl_dirpath, lbl_fname)
        lbl = scipy.io.loadmat(lbl_path)
        train_lbl.append(np.hstack((lbl['box_coord'][0], [lbl_map[d_dirname]])))
        # print("lbl_coord ", lbl['box_coord'])

# cv2.destroyAllWindows()

train_data = np.array(train_data, dtype=np.float64) / 255
train_lbl = np.array(train_lbl, dtype=np.float64)
train_lbl[:, :4] = normalize(train_lbl[:, :4])

print('train_data.shape: ', train_data.shape)
print('train_lbl.shape: ', train_lbl.shape)

plot(train_data, train_lbl[:,4], train_lbl[:,4], w=8, h=7)
# exit()
# -----------------------------------------------------------

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(75, 100, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
# bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid",
	name="bounding_box")(bboxHead)
# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(data_dir), activation="softmax",
	name="class_label")(softmaxHead)

# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
	inputs=vgg.input,
	outputs=(bboxHead, softmaxHead))

# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
losses = {
	"bounding_box": "mean_squared_error",
    "class_label": "sparse_categorical_crossentropy",
}
# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
	"bounding_box": 1.0,
    "class_label": 1.0,
}
# initialize the optimizer, compile the model, and show the model
# summary
model.compile(loss=losses, optimizer=Adam(), metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

# construct a dictionary for our target training outputs
trainTargets = {
	"bounding_box": train_lbl[:, :4],
    "class_label": train_lbl[:, 4],
}
# construct a second dictionary, this one for our target testing
# outputs
# testTargets = {
# 	"class_label": testLabels,
# 	"bounding_box": testBBoxes
# }


# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
H = model.fit(
	train_data, trainTargets,
	# validation_data=(train_data, trainTargets),
	batch_size=50,
	epochs=3,
	verbose=1)


