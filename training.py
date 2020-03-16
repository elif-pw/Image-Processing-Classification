from sklearn.preprocessing import LabelEncoder
import numpy as np
import mahotas
import cv2
import os
import h5py

train_path = "train"
fixed_size = tuple((500, 500))
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
bins = 8

# local feature
def gen_sift_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoint
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


# global feature corner count
def get_corners(image):
    # convert image to gray scale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect corners with the goodFeaturesToTrack function
    feature_params = dict(maxCorners=100, qualityLevel=0.6, minDistance=7, blockSize=7)
    corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    corners = np.int0(corners)
    return len(corners)


# global feature Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# global feature Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# global feature Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


if __name__ == '__main__':
    # img= cv2.imread("isolated\\circinatum\\l01.jpg")
    # fd_hu_moments(img)

    # get the training labels
    train_labels = os.listdir(train_path)
    # sort the training labels
    train_labels.sort()

    # empty lists to hold feature vectors and labels
    global_features = []
    labels = []

    # loop over the training data sub-folders
    for training_name in train_labels:
        dir = train_path + "//" + training_name
        # get the current training label
        current_label = training_name
        num_files = len([f for f in os.listdir(train_path + "//" + training_name) if
                         os.path.isfile(os.path.join(train_path + "//" + training_name, f))])

        # loop over the images in each sub-folder
        for x in range(num_files):
            # get the image file name
            file = dir + "/" + str(x) + ".jpg"

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)

            hu_moments = fd_hu_moments(image)
            haralick = fd_haralick(image)
            histogram = fd_histogram(image)
            corners = get_corners(image)
            kp, desc = gen_sift_features(image)

            resized_desc = np.resize(desc, (1, 200))[0]

            global_feature = np.hstack([hu_moments, haralick, histogram, corners])

            # local_feature=np.hstack(resized_desc)

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)

    # save the feature vector using HDF5
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(global_features))

    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()
