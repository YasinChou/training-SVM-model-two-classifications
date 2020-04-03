import numpy as np
from skimage.feature import hog
import cv2
import time
import pickle
from os import walk
from os import path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



clf_path = 'clf_pickle_all_v1.p'    # if classifier exist
color_space = 'YCrCb'               # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9                          # HOG orientations
pix_per_cell = 8                    # HOG pixels per cell
cell_per_block = 2                  # HOG cells per block, which can handel e.g. shadows
hog_channel = "ALL"                 # Can be 0, 1, 2, or "ALL"
# hog_channel = 2                   # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)             # Spatial binning dimensions
hist_bins = 32                      # Number of histogram bins
spatial_feat = True                 # Spatial features on or off
hist_feat = True                    # Histogram features on or off
hog_feat = True                     # HOG features on or off


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm = 'L1', transform_sqrt=True, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L1', transform_sqrt=True, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel= 'ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel= 'ALL',
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []

        # png is scale from (0,1)
        # image = mpimg.imread(file)
        image = cv2.imread(file)
        image = cv2.resize(image, (300, 300))

        cv2.imshow('image_res:', image)
        cv2.waitKey(2)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def get_fileNames(rootdir): # get all the pictures
    data=[]
    prefix_set =[]
    for root, dirs, files in walk(rootdir, topdown=True):
        for name in files:
            prefix, ending = path.splitext(name)
            if ending != ".jpg" and ending != ".jepg" and ending != ".png":
                continue
            else:
                data.append(path.join(root, name))
                prefix_set.append(prefix)
    return data, prefix_set

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)



def obj_detection(image, svc):
    img = image.copy()
    img = cv2.resize(img, (300, 300))

    X = single_img_features(img,color_space=color_space, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=hog_channel,
                        spatial_feat=True, hist_feat=True, hog_feat=True )

    X = X.reshape(1, -1)
    test_features = X_scaler.transform(X)
    test_prediction = svc.predict(test_features)

    if test_prediction == 1:

        width = int(img.shape[1])
        height = int(img.shape[0])
        cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), 12)
        cv2.putText(img, 'obj_detected', (int(width / 20), int(height / 4)),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.namedWindow('obj_detection_basedon_infrared_image', 0)
        cv2.imshow('obj_detection_basedon_infrared_image', img)
        cv2.waitKey(1000)
    else:
        width = int(img.shape[1])
        height = int(img.shape[0])
        cv2.rectangle(img, (0, 0), (width, height), (0, 255, 0), 12)
        cv2.putText(img, 'NO obj_detected', (int(width / 20), int(height / 4)),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.namedWindow('obj_detection_basedon_infrared_image',0)
        cv2.imshow('obj_detection_basedon_infrared_image', img)
        cv2.waitKey(1000)

    return img


# Load or train the svm, i.e.clf_pickle_all_v1.p
if path.isfile(clf_path):
    print('loading existing classifier...')
    with open(clf_path, 'rb') as file:
        clf_pickle = pickle.load(file)
        svc = clf_pickle["svc"]
        X_scaler = clf_pickle["scaler"]
        orient = clf_pickle["orient"]
        pix_per_cell = clf_pickle["pix_per_cell"]
        cell_per_block = clf_pickle["cell_per_block"]
        spatial_size = clf_pickle["spatial_size"]
        hist_bins = clf_pickle["hist_bins"]
        color_space = clf_pickle["color_space"]
    print('finish loading the classifier.')
else:

    # target and not target file path
    obj_path = '/home/yasin/svm_test/obj'
    not_obj_path = '/home/yasin/svm_test/not_obj'
    obj, _ = get_fileNames(obj_path)
    not_obj, _ = get_fileNames(not_obj_path)

    # set the sample size
    sample_size = min(len(obj), len(not_obj))
    obj = obj[0:sample_size]
    not_obj = not_obj[0:sample_size]

    obj_features = extract_features(obj, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

    none_obj_features = extract_features(not_obj, color_space=color_space,
                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                          orient=orient, pix_per_cell=pix_per_cell,
                                          cell_per_block=cell_per_block,
                                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                                          hist_feat=hist_feat, hog_feat=hog_feat)

    ee = time.time()

    X = np.vstack((obj_features, none_obj_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(obj_features)), np.zeros(len(none_obj_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample


    # save classifier
    clf_pickle = {}
    clf_pickle["svc"] = svc
    clf_pickle["scaler"] = X_scaler
    clf_pickle["orient"] = orient
    clf_pickle["pix_per_cell"] = pix_per_cell
    clf_pickle["cell_per_block"] = cell_per_block
    clf_pickle["spatial_size"] = spatial_size
    clf_pickle["hist_bins"] = hist_bins
    clf_pickle["color_space"] = color_space

    destnation = clf_path
    pickle.dump(clf_pickle, open(destnation, "wb"))
    print("Classifier is written into: {}".format(destnation))



if __name__ == "__main__":

    demo = 1
    # test given image to test if there is obj or not
    if demo == 1:  # single piture

        filename = '/home/yasin/obj/obj_test.jpg'
        image = cv2.imread(filename)

        img_svm = obj_detection(image, svc)

        print('finish the svm result of single image.')

    else:  # pictures flow

        print('calculating the svm result of designated_path...')
        obj_test_path = '/home/yasin/obj/'
        obj_test_images, prefix_set = get_fileNames(obj_test_path)
        cnt_obj = 0
        video_size = (300, 300)
        fps = 1
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        # save the svm result as ouput video
        outVideo_obj = cv2.VideoWriter('./svm_test_result.avi', fourcc, fps, video_size)

        for i in range(len(obj_test_images)):
            image = cv2.imread(obj_test_images[i])
            img_svm = obj_detection(image, svc)
            outVideo_obj.write(img_svm)
        print('TP of the obj_detection_SVM is: ', cnt_obj / len(obj_test_images))  # TP 1
        print('finish the svm result of obj_path.')





