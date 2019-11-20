import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_intrinsic():
    # prepare object points
    nx = 8 # number of inside corners in x
    ny = 6 # number of inside corners in y
    
    # Make a list of calibration images
    fname = 'calibration_test.png'
    img = cv2.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)

def find_homography(points1, points2, method, ransacReprojThreshold=3, mask=None, maxiters=2000, confidence=0.995):
    '''
    :param points1
    :param points2
    :param method
    :param ransacReprojThreshold: Maximum allowed reprojection error to treat a point pair as an inlier
    :param mask
    :param maxiters: The maximum number of RANSAC iterations.
    :param confidence: Confidence level, between 0 and 1.
    '''
    assert len(points1) == len(points2)

    H, mask = cv2.findHomography(points1, points2, method, ransacReprojThreshold, mask, maxiters, confidence)
    return H, mask

def find_essential_matrix(points1, points2, cameraMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0, mask=None):
    '''
    :param points1 
    :param points2
    :param cameraMatrix
    :param method
    :param prob: It specifies a desirable level of confidence that the estimated matrix is correct.
    :param threshold: It is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier.
    :param mask
    '''
    assert len(points1) == len(points2)

    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix, method, prob, threshold, mask)
    return E, mask

def recover_pose(E, points1, points2, cameraMatrix, R=None, t=None, mask=None):
    '''
    :param E: The input essential matrix.
    :param points1
    :param points2
    :param cameraMatrix
    :param R: Output array for recovered relative rotation
    :param t: Output array for recovered relative translation
    :param mask: Input/output mask for inliers in points1 and points2.
    '''
    assert len(points1) == len(points2)

    points, R, t, mask = cv2.recoverPose(E, points1, points2, cameraMatrix, mask)
    return points, R, t, mask

def triangulate_points(projMatr1, projMatr2, projPoints1, projPoints2, points4D=None):
    '''
    :param projMatr1: 3x4 projection matrix of the first camera.
    :param projMatr2: 3x4 projection matrix of the second camera.
    :param projPoints1: 2xN array of feature points in the first image
    :param projPoints2: 2xN array of corresponding points in the second image.
    :param points4D: output array of reconstructed points in homogeneous coordinates.
    '''
    assert len(projPoints1) == len(projPoints2)

    points4D = cv2.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2, points4D)
    return points4D