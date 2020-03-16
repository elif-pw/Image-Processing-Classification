import os
import training
import h5py
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
import glob
import cv2
from sklearn.model_selection import train_test_split, cross_val_score, KFold

warnings.filterwarnings('ignore')

fixed_size = tuple((500, 500))
num_trees = 100
test_size = 0.10
seed = 9
train_path = "train"
test_path = "test"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
scoring = "accuracy"

model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# variables to hold the results and names
results = []
names = []

# import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = \
    train_test_split(np.array(global_features),
                     np.array(global_labels),
                     test_size=test_size,
                     random_state=seed)

# 10-fold cross validation
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
total = "%s: %f " % ("RF", cv_results.mean())
print(total)

# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

train_labels = os.listdir(train_path)
print(train_labels)

count = 0
success = 0
# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)
    # resize
    image = cv2.resize(image, fixed_size)
    # get the global features
    hu_moments = training.fd_hu_moments(image)
    haralick = training.fd_haralick(image)
    histogram = training.fd_histogram(image)
    corners = training.get_corners(image)
    # get a local feature
    kp, desc = training.gen_sift_features(image)
    resized = np.resize(desc, (1, 200))[0]

    global_feature = np.hstack([hu_moments, haralick, histogram, corners])
    # local_feature=np.hpstack(resized)

    prediction = clf.predict([global_feature])[0]

    # simple testing of the predictions
    if count < 14:
        if prediction == 0:
            success = success + 1
    if 13 < count < 31:
        if prediction == 1:
            success = success + 1
    if 30 < count < 46:
        if prediction == 2:
            success = success + 1
    if 45 < count < 66:
        if prediction == 3:
            success = success + 1
    if 65 < count < 83:
        if prediction == 4:
            success = success + 1
    else:
        if prediction == 5:
            success = success + 1
    count = count + 1

print("Success rate:  " + str(success / count))
