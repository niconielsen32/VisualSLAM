import bz2
import os
from matplotlib.cbook import normalize_kwargs
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
from lib.visualization.plotting import plot_residual_results, plot_sparsity

def read_bal_data(file_name):
    """
    Loads the data

    Parameters
    ----------
    file_name (str): The file path for the data

    Returns
    -------
    cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
    Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
    cam_idxs (ndarray): Shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
    Q_idxs (ndarray): Shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
    qs (ndarray): Shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.
    """
    with open(file_name, "rt") as file:
        n_cams, n_Qs, n_qs = map(int, file.readline().split())

        cam_idxs = np.empty(n_qs, dtype=int)
        Q_idxs = np.empty(n_qs, dtype=int)
        qs = np.empty((n_qs, 2))

        for i in range(n_qs):
            cam_idx, Q_idx, x, y = file.readline().split()
            cam_idxs[i] = int(cam_idx)
            Q_idxs[i] = int(Q_idx)
            qs[i] = [float(x), float(y)]

        cam_params = np.empty(n_cams * 9)
        for i in range(n_cams * 9):
            cam_params[i] = float(file.readline())
        cam_params = cam_params.reshape((n_cams, -1))

        Qs = np.empty(n_Qs * 3)
        for i in range(n_Qs * 3):
            Qs[i] = float(file.readline())

        Qs = Qs.reshape((n_Qs, -1))
    
    return cam_params, Qs, cam_idxs, Q_idxs, qs


def reindex(idxs):
    keys = np.sort(np.unique(idxs))
    key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}
    return [key_dict[idx] for idx in idxs]


def shrink_problem(n, cam_params, Qs, cam_idxs, Q_idxs, qs):
    """
    Shrinks the problem to be n points

    Parameters
    ----------
    n (int): Number of points the shrink problem should be
    cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
    Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
    cam_idxs (ndarray): Shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
    Q_idxs (ndarray): Shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
    qs (ndarray): Shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.

    Returns
    -------
    cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
    Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
    cam_idxs (ndarray): Shape (n,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
    Q_idxs (ndarray): Shape (n,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
    qs (ndarray): Shape (n, 2) contains measured 2-D coordinates of points projected on images in each observations.
    """
    cam_idxs = cam_idxs[:n]
    Q_idxs = Q_idxs[:n]
    qs = qs[:n]
    cam_params = cam_params[np.isin(np.arange(cam_params.shape[0]), cam_idxs)]
    Qs = Qs[np.isin(np.arange(Qs.shape[0]), Q_idxs)]

    cam_idxs = reindex(cam_idxs)
    Q_idxs = reindex(Q_idxs)
    return cam_params, Qs, cam_idxs, Q_idxs, qs


def rotate(Qs, rot_vecs):
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.

    Parameters
    ----------
    Qs (ndarray): The 3D points
    rot_vecs (ndarray): The rotation vectors

    Returns
    -------
    Qs_rot (ndarray): The rotated 3D points
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v


def project(Qs, cam_params):
    """
    Convert 3-D points to 2-D by projecting onto images.

    Parameters
    ----------
    Qs (ndarray): The 3D points
    cam_params (ndarray): Initial parameters for cameras

    Returns
    -------
    qs_proj (ndarray): The projectet 2D points
    """
    # Rotate the points
    qs_proj = rotate(Qs, cam_params[:, :3])
    # Translat the points
    qs_proj += cam_params[:, 3:6]
    # Un-homogenized the points
    qs_proj = -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
    # Distortion
    f, k1, k2 = cam_params[:, 6:].T
    n = np.sum(qs_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    qs_proj *= (r * f)[:, np.newaxis]
    return qs_proj


def objective(params, cam_param, n_cams, n_Qs, cam_idxs, Q_idxs, qs):
    """
    The objective function for the bundle adjustment

    Parameters
    ----------
    params (ndarray): Camera parameters and 3-D coordinates.
    n_cams (int): Number of cameras
    n_Qs (int): Number of points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points
    qs (ndarray): The image points

    Returns
    -------
    residuals (ndarray): The residuals
    """
    # Should return the residuals consisting of the difference between the observations qs and the reporjected points
    # Params is passed from bundle_adjustment() and contains the camera parameters and 3D points
    # project() expects an arrays of shape (len(qs), 3) indexed using Q_idxs and (len(qs), 9) indexed using cam_idxs
    # Copy the elements of the camera parameters and 3D points based on cam_idxs and Q_idxs

    # Get the camera parameters
    # cam_params = params[:n_cams * 9].reshape((n_cams, 9))
    K = np.array((718.856,0,0),(0,718.856,0),(0,0,1))
    
    

    cam_params = cam_param.reshape((n_cams,9))

    # Get the 3D points
    Qs = params.reshape((n_Qs, 3))
    # Qs = params[n_cams * 9:].reshape((n_Qs, 3))

    # Project the 3D points into the image planes
    qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])

    # Calculate the residuals
    residuals = (qs_proj - qs).ravel()

    #q = K* [R t] * Qs
    return residuals


def bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs):
    """
    Preforms bundle adjustment

    Parameters
    ----------
    cam_params (ndarray): Initial parameters for cameras
    Qs (ndarray): The 3D points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points
    qs (ndarray): The image points

    Returns
    -------
    residual_init (ndarray): Initial residuals
    residuals_solu (ndarray): Residuals at the solution
    solu (ndarray): Solution
    """
    # Use least_squares() from scipy.optimize to minimize the objective function
    # Stack cam_params and Qs after using ravel() on them to create a one dimensional array of the parameters
    # save the initial residuals by manually calling the objective function
    # residual_init = objective()
    # res = least_squares(.....)

    # Stack the camera parameters and the 3D points
    params = np.hstack((cam_params.ravel(), Qs.ravel()))

    # Save the initial residuals
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

    # Perform the least_squares optimization
    res = least_squares(objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf', max_nfev=50,
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    # normalized_cost = res.cost / res.x.size()
    # print ("\nNormalized cost with reduced points: " +  str(normalized_cost))
    return residual_init, residuals_solu, solu

def sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs):
    """
    Create the sparsity matrix

    Parameters
    ----------
    n_cams (int): Number of cameras
    n_Qs (int): Number of points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points

    Returns
    -------
    sparse_mat (ndarray): The sparsity matrix
    """
    
    # m = cam_idxs.size * 2  # number of residuals
    # n = n_cams * 9 + n_Qs * 3  # number of parameters
    # print("m:\n" + str(m) + "\nn:\n" + str(n))
    # sparse_mat = lil_matrix((m, n), dtype=int)
    # # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

    # i = np.arange(cam_idxs.size)
    # # Sparsity from camera parameters
    # for s in range(9):
    #     sparse_mat[2 * i, cam_idxs * 9 + s] = 1
    #     sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1
    # #print (sparse_mat)
    # # Sparsity from 3D points
    # for s in range(3):
    #     sparse_mat[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
    #     sparse_mat[2 * i + 1, n_cams * 9 + Q_idxs * 3 + s] = 1


    m = cam_idxs.size * 2  # number of residuals
    n = n_Qs * 3  # number of parameters
    print("m:\n" + str(m) + "\nn:\n" + str(n))
    sparse_mat = lil_matrix((m, n), dtype=int)
    # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

    i = np.arange(cam_idxs.size)
    # Sparsity from camera parameters
    for s in range(9):
        sparse_mat[2 * i, cam_idxs * 9 + s] = 1
        sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1
    #print (sparse_mat)
    # Sparsity from 3D points
    for s in range(3):
        sparse_mat[2 * i, Q_idxs * 3 + s] = 1
        sparse_mat[2 * i + 1,  Q_idxs * 3 + s] = 1



    return sparse_mat


def bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat):
    """
    Preforms bundle adjustment with sparsity

    Parameters
    ----------
    cam_params (ndarray): Initial parameters for cameras
    Qs (ndarray): The 3D points
    cam_idxs (list): Indices of cameras for image points
    Q_idxs (list): Indices of 3D points for image points
    qs (ndarray): The image points
    sparse_mat (ndarray): The sparsity matrix

    Returns
    -------
    residual_init (ndarray): Initial residuals
    residuals_solu (ndarray): Residuals at the solution
    solu (ndarray): Solution
    """

    transformations = []
    for i in range(len(cam_params)):
        R ,_ = cv2.Rodrigues( cam_params[:3])
        t = cam_params[3:6]
        transformations.append(np.column_stack((R,t)))

    # Stack the camera parameters and the 3D points
    params = np.hstack((cam_params.ravel(), Qs.ravel()))
    params2 = Qs.ravel()

    # Save the initial residuals
    residual_init = objective(params2, cam_params.ravel() , cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

    # Perform the least_squares optimization with sparsity
    res = least_squares(objective, params2, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-6, method='trf', max_nfev=50,
                        args=(cam_params.ravel(), cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    # normalized_cost = res.cost / res.x.size()
    # print ("\nAverage cost for each point (solution with sparsity): " +  str(normalized_cost))
    return residual_init, residuals_solu, solu


def run_BA():
    data_file = 'b_adj.txt'

    cam_params, Qs, cam_idxs, Q_idxs, qs = read_bal_data(data_file)
    # cam_params_small, Qs_small, cam_idxs_small, Q_idxs_small, qs_small = shrink_problem(1000, cam_params, Qs, cam_idxs,
    #                                                                                     Q_idxs, qs)

    # n_cams_small = cam_params_small.shape[0]
    # n_Qs_small = Qs_small.shape[0]
    # print("n_cameras: {}".format(n_cams_small))
    # print("n_points: {}".format(n_Qs_small))
    # print("Total number of parameters: {}".format(9 * n_cams_small + 3 * n_Qs_small))
    # print("Total number of residuals: {}".format(2 * qs_small.shape[0]))

    # small_residual_init, small_residual_minimized, opt_params = bundle_adjustment(cam_params_small, Qs_small,
    #                                                                               cam_idxs_small,
    #                                                                               Q_idxs_small, qs_small)

    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(9 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat)



    plot_residual_results(qs, residual_init, residual_minimized)
    
    return opt_params



def main():
    #data_file = os.path.join("problem-49-7776-pre", "problem-49-7776-pre.txt.bz2")
    #data_file = 'b_adj.txt'
    data_file = 'bundle_data.txt'
    cam_params, Qs, cam_idxs, Q_idxs, qs = read_bal_data(data_file)
    cam_params_small, Qs_small, cam_idxs_small, Q_idxs_small, qs_small = shrink_problem(1000, cam_params, Qs, cam_idxs,
                                                                                        Q_idxs, qs)

    n_cams_small = cam_params_small.shape[0]
    n_Qs_small = Qs_small.shape[0]
    print("n_cameras: {}".format(n_cams_small))
    print("n_points: {}".format(n_Qs_small))
    print("Total number of parameters: {}".format(9 * n_cams_small + 3 * n_Qs_small))
    print("Total number of residuals: {}".format(2 * qs_small.shape[0]))

    small_residual_init, small_residual_minimized, opt_params = bundle_adjustment(cam_params_small, Qs_small,
                                                                                  cam_idxs_small,
                                                                                  Q_idxs_small, qs_small)

    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(9 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat)
   

    plot_residual_results(qs_small, small_residual_init, small_residual_minimized, qs, residual_init,
                          residual_minimized)


if __name__ == "__main__":
    main()
