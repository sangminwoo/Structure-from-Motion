import os
import argparse
import cv2
import numpy as np

from utils import *
from structure import *

def sift(imGray):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(imGray, None)
    print(f'keypoints: {len(keypoints)}, descriptors: {descriptors.shape}')
    return keypoints, descriptors

def create_matcher(trees, checks):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
    search_params = dict(checks=checks)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return matcher

def find_good_matches(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor=0.65):
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    points1 = []
    points2 = []
    
    for m, n in matches:
        if m.distance < factor * n.distance:
            good_matches.append(m)
            points1.append(keypoints1[m.queryIdx].pt)
            points2.append(keypoints2[m.trainIdx].pt)

    #src_points = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points1 = np.asarray(points1)
    #dst_points = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points2 = np.asarray(points2)

    #H, mask = findHomography(src_points, dst_points, cv2.RANSAC, 1.0)
    #F, mask = cv2.findFundamentalMat(src_points, dst_points, cv2.RANSAC)
    #E, mask = find_essential_matrix(src_points, dst_points, K, cv2.RANSAC)

    #points1 = src_points[mask.ravel() == 1]
    #points2 = dst_points[mask.ravel() == 1]

    return good_matches, points1, points2 #, points1.T, points2.T



'''
close all;
clear all;

addpath('Givenfunctions');

%% Define constants and parameters
% Constants ( need to be set )
number_of_iterations_for_5_point    = 0;

% Thresholds ( need to be set )
threshold_of_distance = 0; 

% Matrices
K               = [ 1698.873755 0.000000     971.7497705;
                    0.000000    1698.8796645 647.7488275;
                    0.000000    0.000000     1.000000 ];

%% Feature extraction and matching
% Load images and extract features and find correspondences.
% Fill num_Feature, Feature, Descriptor, num_Match and Match
% hints : use vl_sift to extract features and get the descriptors.
%        use vl_ubcmatch to find corresponding matches between two feature sets.



%% Initialization step
% Estimate E using 8,7-point algorithm or calibrated 5-point algorithm and RANSAC
E; % find out

% Decompose E into [R, T]
R; % find out
T; % find out

% Reconstruct 3D points using triangulation
X; % find out
X_with_color; % [6 x # of feature matrix] - XYZRGB

% Save 3D points to PLY
SavePLY('2_views.ply', X_with_color);
'''