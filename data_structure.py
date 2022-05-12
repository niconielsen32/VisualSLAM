
import cv2
from matplotlib import projections

import numpy as np 
from numpy.linalg import inv
import os
import matplotlib.pyplot as plt

class node ():
    def __init__(self, keypoints=None, frameID=None, parrentID=None, img=None) -> None:
        self.points2D = keypoints
        self.points3D = None
        self.keypoints = keypoints
        self.transformation = None
        self.frameID = frameID
        self.indexID = None
        self.parrent = parrentID
        self.image = img


    def set_transform(self, transform):
        self.transformation = transform
    
    def get_2D_points(self):
        return self.points2D

    def get_3D_points(self):
        return self.points3D

class graph():
    def __init__(self) -> None:
        self.data = []
        self._3DPoints = []
        self.points_to_plot = []
        self.num_2DPoints = 0
        
        self.output = [] # cam_idx, 3D idx, 2d point

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        self.current_pose = None
        # To clear the file if it exists.
        open('b_adj.txt', 'w').close()


    def add_node(self, node):
        self.data.append(node)

    def create_data_set(self, node ):
        f = open("b_adj.txt", 'a')
        
        num_points = len(node.points2D)
        start_index_3D_point = len(self._3DPoints) - num_points

        for point in node.points2D:
            save_point = point.flatten()
            data_to_write = str(node.frameID - 1) + " " + str(start_index_3D_point) + " " + str(point[0]) + " " + str(point[1])+ '\n'
            start_index_3D_point += 1
            f.write(data_to_write)
        
        f.close()

    def save_point(self, node, _2dpoint, idx_3dpoint):
        f = open('b_adj.txt', 'a')
        line_to_write = str(node.frameID - 1) + " " + str(idx_3dpoint) + " " + str(_2dpoint[0]) + " "  + str(_2dpoint[1]) + '\n'
        f.write(line_to_write)
        f.close()

    def save_3D_points(self):
        print("Adding the 3D points...\n")
        f = open('b_adj.txt', 'a')
        for point in self._3DPoints:
            lines_to_write = str(point[0]) + '\n' + str(point[1]) + '\n' + str(point[2]) + '\n'
            f.write(lines_to_write)
        f.close()

    def track_3D_points(self, max_error=4):
        print ("Tracking 3D points...\n")
        for i in range(len(self.data)):
            self.current_pose = np.matmul(self.current_pose, self.data[i].transformation)
            print ("\nCurrent pose:\n " + str (self.current_pose))
            
            
            _3Dpoints = self.data[i].get_3D_points() # Returns a list of 3D points
            for point in _3Dpoints:
                self.points_to_plot.append(point)
            _3Dpoints = np.array(_3Dpoints)


            tmp_list = []
            for point in _3Dpoints:
                homogen_point = np.append(point,1)
                newPoint = np.matmul(self.current_pose, homogen_point )
                tmp_list.append(newPoint[:3])
            _3Dpoints = tmp_list

            index_list_3D_point = []
            current_list_size = len(self._3DPoints)
            # Append each point from the list _3Dpoints to an other list containing all the 3D points
            # from all the frames. 
            for p in _3Dpoints:
                index_list_3D_point.append(current_list_size)
                self._3DPoints.append(p)
                current_list_size += 1

            kp1 = self.data[i].get_2D_points()
            img1 = self.data[i].image

            self.create_data_set(self.data[i])
            self.num_2DPoints += len(kp1)
            # print(index_list_3D_point)
            # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
            trackpoints1 = np.expand_dims(kp1, axis=1)
            # for j in range(i + 1, len(self.data)):
            #     if (j > i + 3 ):
            #         break
                
            #     trackpoints1 = np.expand_dims(kp1, axis=1)
            #     points_to_save = _3Dpoints

        
            #     img2 = self.data[j].image
 
            #     # Use optical flow to find tracked counterparts
            #     trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

            #     # Convert the status vector to boolean so we can use it as a mask
            #     trackable = st.astype(bool)
            #     # print("Trackable: " + str(trackable.shape))


            #     for k in range(len(trackpoints2)):
            #         h, w = img1.shape
            #         #print(trackpoints2[k][0][1])
            #         if st[k] and err[k] < max_error and trackpoints2[k][0][0] < w and trackpoints2[k][0][1] < h :
            #             _2Dpoint = trackpoints2[k][0][0], trackpoints2[k][0][1]
            #             self.save_point(self.data[j], _2Dpoint, index_list_3D_point[k] )
            #             self.num_2DPoints += 1

            #     # Create a maks there selects the keypoints there was trackable and under the max error
            #     under_thresh = np.where(err[trackable] < max_error, True, False)

            #     # Use the mask to select the keypoints
            #     trackpoints1 = trackpoints1[trackable][under_thresh]
            #     trackpoints2 = np.around(trackpoints2[trackable][under_thresh])
                


            #     # Remove the keypoints there is outside the image
            #     h, w = img1.shape
            #     in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
                
            #     trackpoints1 = trackpoints1[in_bounds]
            #     trackpoints2 = trackpoints2[in_bounds]

                
            #     del(under_thresh)
              

    def add_transforms(self):
        print("Adding transformations...\n")
        f = open('b_adj.txt', 'a')
        focal_length = '718.856'
        distortion = '0'
        for node in self.data:
            R = node.transformation[0:3,0:3]
            # print ("R before Rod")
            # print(R)
            
            R,_ = cv2.Rodrigues(R)
            R = R.flatten()
            # print (R)
            # print ()
            t = node.transformation[0:3,3]
            

            line_to_write = str(R[0]) + '\n' + str(R[1]) + '\n' + str(R[2]) + '\n'
            line_to_write += str(t[0]) + '\n' + str(t[1]) + '\n' + str(t[2]) + '\n'
            line_to_write += focal_length + '\n' + distortion + '\n' + distortion + '\n'
            f.write(line_to_write)
        f.close()

    def add_start_of_document(self):
        print ("Writing the header of the file...\n")
        values = str(len(self.data)) + " " + str(len(self._3DPoints)) + " " + str(self.num_2DPoints)

        self.prepend_line('b_adj.txt', values)

    def __str__(self) -> str:
        output = ''
        for n in self.data:
            output += str(n.frameID) + " Length of 3D points: " + str(len(n.points3D)) + " len 2D: " + str(len(n.keypoints)) + '\n'
        return output

    def prepend_line(self, file_name, line):
        dummy_file = file_name + '.bak'
        with open(file_name, 'r') as read_obj, open(dummy_file,'w') as write_obj:
            write_obj.write(line + '\n')
            for line in read_obj:
                write_obj.write(line)
            
        os.remove(file_name)
        os.rename(dummy_file, file_name)

    def plot_3D_points(self):
        fig = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        
        ax = fig.add_subplot(projection='3d')
        ax2 =fig2.add_subplot()
        ax3 =fig3.add_subplot()

        x = []
        y = []
        z = []
        count = 0
        for point in self._3DPoints:
            if count % 200 == 0:
                # if point[2] < 100 :  
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
            count += 1
        print (f"Number of points shown is: {len(x)}" )
        ax.scatter(x,y,z)
        ax.set_title("Global 3D Points")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        x2 = []
        y2 = []
        z2 = []
        count = 0
        for point in self.points_to_plot:
            if count % 200 == 0:
                # if point[2] < 50:  
                x2.append(point[0])
                y2.append(point[1])
                z2.append(point[2])
            count += 1
        print (f"Number of points shown is: {len(x)}" )
        ax2.scatter(x2,y2)
        ax2.set_title("Relative 3D points")
        ax2.set_xlabel('X Label')
        ax2.set_ylabel('Y Label')
        # ax2.set_zlabel('Z Label')

        
        x3 = []
        y3 = []
        z3 = []
        start_pose = self.data[0].transformation
        for i in range(len(self.data)-1):
            x3.append(start_pose[0,3])
            y3.append(start_pose[1,3])
            z3.append(start_pose[2,3])
            start_pose = np.matmul(start_pose, self.data[i+1].transformation )

        ax3.scatter(x3,z3, c='b')
        ax3.scatter(x,z, c='r')
        ax3.set_xlabel('X Label')
        ax3.set_ylabel('Z Label')
        # ax3.set_zlabel('Z Label')

        plt.show()