import os
import numpy as np
import cv2
from pandas import describe_option
from scipy.optimize import least_squares
from sklearn.feature_extraction.text import TfidfVectorizer

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm
from pprint import pprint

import bag_of_words

import dbow


class VisualOdometry():
    def __init__(self, data_dir):
        
        print("Loading Images")
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib("C:/Users/45213/Downloads/data_odometry_gray/dataset/sequences/07/calib.txt")
        self.gt_poses = self._load_poses("C:/Users/45213/Downloads/data_odometry_poses/dataset/poses/07.txt")
        #self.images_l = self._load_images("C:/Users/45213/OneDrive/Desktop/sequence3/images_l")
        #self.images_r = self._load_images("C:/Users/45213/OneDrive/Desktop/sequence3/images_r")
        self.images_l = self._load_images("C:/Users/45213/Downloads/data_odometry_gray/dataset/sequences/07/image_0")
        self.images_r = self._load_images("C:/Users/45213/Downloads/data_odometry_gray/dataset/sequences/07/image_1")

        print("Number of left images: ", len(self.images_l))
        print("Number of right images: ", len(self.images_r))
        print("Number of poses: ", len(self.gt_poses))

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

                # Create Vocabulary
        n_clusters = 10
        depth = 2
        #vocabulary = dbow.Vocabulary([], n_clusters, depth)
        # Loading the vocabulary
        #print("Loading Vocabulary")
        #self.vocabulary = dbow.Vocabulary.load('KITTIORB.pickle')

        # Create a database
        #self.db = dbow.Database(self.vocabulary)

        #self.mean_scores = []


        # ORB
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)



    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def detect_orb_keypoints(self, img1, img2):
        
        keypoints1, descriptors1 = self.orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(img2, None)

        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        q1 = [keypoints1[m.queryIdx] for m in good]
        #q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])

        return q1, descriptors1
    


    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2


    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image

        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix
    
    def loop_detection(self, descriptors):
        
        number_loop_frames = 5
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descriptors]
        self.db.add(descs)
        scores = self.db.query(descs)
        print("Scores size: ", len(scores))
        
        if len(self.mean_scores) >= 2:
            print(self.mean_scores)
            #sorted_scores = np.sort(np.array(self.mean_scores))
            #print(sorted_scores)

        if len(scores) > 1:
            index_similar_image = np.argmax(scores[:len(scores)-1])
            if scores[index_similar_image] > 0.9:
                print(index_similar_image)
                print(scores[index_similar_image])


    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Get teh tiled keypoints
        kp1_l, descriptors = self.detect_orb_keypoints(img1_l, img2_l)

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        if len(tp1_r) and len(tp2_r):
            # Calculate the 3D points
            Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

            # Estimate the transformation matrix
            transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)

            #self.loop_detection(descriptors)
        
            return transformation_matrix, descriptors
        return None, descriptors


def main():
    data_dir = 'KITTI_sequence_1'  # Try KITTI_sequence_2
    vo = VisualOdometry(data_dir)

    #play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip

    n_clusters = 10 
    n_features = 200
    assert n_clusters != 0 and n_features != 0, "Remember to change n_clusters and n_features in main"

    #print("Training bag of words")
    # Make the BoW and train it on the training data
    bow = bag_of_words.BoW(n_clusters, n_features)
    #bow.train(vo.images_l)
    #print("Done training bag of words")

   
    keyframe_size = 100
    frame_count = 0
    number_of_loops_detected = 0

    gt_path = []
    estimated_path = []
    
    traj = np.zeros(shape=(600, 800, 3))

    for i in tqdm(range(len(vo.images_l))):
        gt_pose = vo.gt_poses[i]
        if i < 1:
            cur_pose = gt_pose
        else:
            frame_count += 1

        
            transf, descriptors = vo.get_pose(i)
            # Find the closest match in the training set
            
            idx = bow.predict(vo.images_l[i], keyframe_size, frame_count, descriptors)
            if idx != -1:
                frame_count = 0
                number_of_loops_detected += 1
                print("Number of loops detected: ", number_of_loops_detected)
                # If a match was found make a show_image with the query and match image
                show_image = bag_of_words.make_stackimage(vo.images_l[i], vo.images_l[idx])
                print("id query: ", i)

                gt_pose_x_loop, gt_pose_y_loop = cur_pose[0, 3], cur_pose[2, 3]
                traj = cv2.circle(traj, (int(gt_pose_x_loop) + 500, int(gt_pose_y_loop) + 300), 12, list((255, 0, 0)), 2)
                # Show the result
                bag_of_words.put_text(show_image, "bottom_center", f"Press any key.. ({i}/{len(vo.images_l)}). ESC to stop")
                cv2.imshow("match", show_image)
                key = cv2.waitKey(0)
                if key == 27:
                    break


            if transf is not None:
                cur_pose = np.matmul(cur_pose, transf)

        gt_pose_x, gt_pose_y = gt_pose[0, 3], gt_pose[2, 3]
        estimated_pose_x, estimated_pose_y = cur_pose[0, 3], cur_pose[2, 3]


        traj = cv2.circle(traj, (int(gt_pose_x) + 500, int(gt_pose_y) + 300), 1, list((0, 0, 255)), 4)
        traj = cv2.circle(traj, (int(estimated_pose_x) + 500, int(estimated_pose_y) + 300), 1, list((0, 255, 0)), 4)

        cv2.putText(traj, 'Actual Position:', (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
        cv2.putText(traj, 'Red', (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
        cv2.putText(traj, 'Estimated Odometry Position:', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
        cv2.putText(traj, 'Green', (270, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.putText(traj, 'Loop Detected:', (145, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
        cv2.putText(traj, 'Blue', (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)

        cv2.imshow('trajectory', traj)
        cv2.waitKey(1)
    
    cv2.imwrite("finalPath.png", traj)
        #gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        #estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    #plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
               #              file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
