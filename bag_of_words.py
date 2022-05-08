from cProfile import label
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

from lib.visualization.image import put_text
from sklearn.feature_extraction.text import TfidfVectorizer


class BoW:
    def __init__(self, n_clusters, n_features):
        # Create the ORB detector
        self.extractor = cv2.ORB_create(nfeatures=n_features)
        self.n_clusters = n_clusters
        # Make a kmeans cluster
        self.kmeans = KMeans(self.n_clusters, verbose=0)

        self.database = []
        self.N_i = [0]*n_clusters
        self.N = 0
        self.tf_idf_weights_list = []
        self.tf_idf_means = []

        self.score_threshold = 12.0
        self.consecutive_frames_under_threshold = 0
        self.frames_under_threshold = 2

        
        with open("BagOfWordsKmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)

        with open("tf_idf_means.pkl", "rb") as f:
            self.tf_idf_means = pickle.load(f)
        

    def train(self, imgs):
        """
        Make the bag of "words"

        Parameters
        ----------
        imgs (list): A list with training images. Shape (n_images)
        """
        self.N = len(imgs)
        print(self.N)
        # Detect ORB features for the images
        _, dlist = zip(*[self.extractor.detectAndCompute(img, None) for img in tqdm(imgs,
                                                                                    desc='Computing local descriptors',
                                                                                    unit=" image")])
        dpool = []
        no_feature = []
        for i, d in enumerate(dlist):
            if d is None:
                no_feature.append(i)
            else:
                dpool += list(d)
        dpool = np.array(dpool)

        print("Making word clusters")
        # Make the "word" clusters
        self.kmeans = self.kmeans.fit(dpool)

        print("Making bag of words")
        # Make the bag of "words"
        dlist = [d for i, d in enumerate(dlist) if i not in no_feature]
        self.db = [self.hist(d) for d in dlist]

        self.tf_idf_means = np.mean(self.tf_idf_weights_list, axis=0)

        with open("tf_idf_means.pkl", "wb") as f:
            pickle.dump(self.tf_idf_means, f)

        with open("BagOfWordsKmeans2.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

        print("Saved")

    def hist(self, descriptors, predict=False):
        """
        Make the histogram for words in the descriptors

        Parameters
        ----------
        descriptors (ndarray): The ORB descriptors. Shape (n_features, 32)

        Returns
        -------
        hist (ndarray): The histogram. Shape (n_clusters)
        """
        # Get the clusters for the descriptors
        labels = self.kmeans.predict(descriptors)
        # Make a histogram over which words there was present in the image
        if predict:
            hist, bin_values = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters - 1))
            return np.multiply(hist, self.tf_idf_means)

        if not predict:

            hist, bin_values = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters - 1))

            sum_of_hist_values = np.sum(hist)

            hist_values_normalized = []
            for component in hist:
                hist_values_normalized.append(component / sum_of_hist_values)

            self.N_i = np.add(self.N_i, hist)

            tf_idf_weights = np.multiply(hist_values_normalized, np.log(self.N / self.N_i))

            self.tf_idf_weights_list.append(tf_idf_weights)

        return hist


    def predict(self, img, keyframe_size, keyframe, d):
        """
        Finds the closest match in the training set to the given image

        Parameters
        ----------
        img (ndarray): The query image. Shape (height, width [,3])

        Returns
        -------
        match_idx (int): The index of the training image there was closest, -1 if there was non descriptors in the image
        """

        def chi2(x, y):
            return np.sum(2 * (x - y) ** 2 / np.maximum(1,(x + y)))

        def euclidean(x,y):
            print(x,y)
            diff = np.subtract(x,y)
            return diff.sum()

        # Get the descriptors from the image
        #_, d = self.extractor.detectAndCompute(img, None)


        # If there was descriptors in the image
        if d is not None:
            # Get the histogram of the words in the image
            # Get the clusters for the descriptors
            h = self.hist(d, predict=True)
        

            self.database.append(h)
            
            if len(self.database) > keyframe_size + 1:
                # Find the training image with the shortest distance
                dist = [chi2(h, entry) for entry in self.database[:-keyframe_size]]
                match_idx = np.argmin(dist)
                #print("Score: ", dist[match_idx])
                if dist[match_idx] < self.score_threshold:
                    self.consecutive_frames_under_threshold += 1
                    print("score: ", dist[match_idx])
                    if self.consecutive_frames_under_threshold >= self.frames_under_threshold:
                        self.consecutive_frames_under_threshold = 0
                        print("\n")
                        print("score con: ", dist[match_idx])
                        print("id match: ", match_idx)
                        return match_idx

                else:
                    self.consecutive_frames_under_threshold = 0
                
        
        return -1


def split_data(dataset, test_size=0.1):
    """
    Loads the images and split it into a train and test set

    Parameters
    ----------
    dataset (str): The path to the dataset
    test_size (float): Represent the proportion of the dataset to include in the test split

    Returns
    -------
    train_img (list): The images in the training set. Shape (n_images)
    test_img (list): The images in the test set. Shape (n_images)
    """
    # Load the iamges
    images = [os.path.join(dataset, image) for image in os.listdir(dataset)]
    images = [cv2.imread(image) for image in tqdm(images, desc='Loading dataset', unit=" image")]

    # Split the data
    train_img, test_img = train_test_split(images, test_size=test_size)
    return train_img, test_img


def make_stackimage(query_image, match_image=None):
    """
    hstack the query and match image

    Parameters
    ----------
    query_image (ndarray): The query image. Shape (height, width [,3])
    match_image (ndarray): The match image. Shape (height, width [,3])

    Returns
    -------
    stack_image (ndarray): The stack image. Shape (height, 2*width [,3])
    """
    match_found = True
    if match_image is None:
        match_image = np.zeros_like(query_image)
        match_found = False

    if len(query_image.shape) != len(match_image.shape):
        if len(query_image.shape) != 3:
            query_image = cv2.cvtColor(cv2.COLOR_GRAY2BGR, query_image)
        if len(match_image.shape) != 3:
            match_image = cv2.cvtColor(cv2.COLOR_GRAY2BGR, match_image)

    height1, width1, *_ = query_image.shape
    height2, width2, *_ = match_image.shape
    height = max([height1, height2])
    width = max([width1, width2])

    if len(query_image.shape) == 2:
        stack_shape = (height, width * 2)
    else:
        stack_shape = (height, width * 2, 3)

    if match_found:
        put_text(query_image, "top_center", "Query")
        put_text(match_image, "top_center", "Match")

    stack_image = np.zeros(stack_shape, dtype=match_image.dtype)
    stack_image[0:height1, 0:width1] = query_image
    stack_image[height1:height1 + height2, 0:width2] = match_image

    if not match_found:
        put_text(stack_image, "top_center", "No features found")

    return stack_image


"""if __name__ == "__main__":
    dataset = 'C:/Users/45213/OneDrive/Desktop/bow_dataset'  # '../data/COIL20/images' # '../data/StanfordDogs/images'
    n_clusters = 10 
    n_features = 200
    assert n_clusters != 0 and n_features != 0, "Remember to change n_clusters and n_features in main"

    # Split the data
    train_img, test_img = split_data(dataset, test_size=0.1)

    # Make the BoW and train it on the training dataq
    bow = BoW(n_clusters, n_features)
    print(bow)
    bow.train(train_img)

    win_name = "query | match"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1024, 600)

    # Find matches to every test image
    for i, img in enumerate(test_img):
        # Find the closest match in the training set
        idx = bow.predict(img)
        if idx != -1:
            # If a match was found make a show_image with the query and match image
            show_image = make_stackimage(img, train_img[idx])
        else:
            # If a match was not found make a show_image with the query image
            print("No features found")
            show_image = make_stackimage(img)

        # Show the result
        put_text(show_image, "bottom_center", f"Press any key.. ({i}/{len(test_img)}). ESC to stop")
        cv2.imshow(win_name, show_image)
        key = cv2.waitKey()
        if key == 27:
            break"""
