import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from feature_matching import *
from structure import *
from utils import *

def get_arguments():
    parser = argparse.ArgumentParser(description='Structure from Motion')
    parser.add_argument('-r', '--read_dir', type=str, default='data/', help='read directory')
    parser.add_argument('-s', '--save_dir', type=str, default='save/', help='save directory')
    parser.add_argument('-f', '--factor', type=float, default='0.7', help='feature matching factor; the higher, the more outliers')
    return parser.parse_args()

def main():
    arg = get_arguments()

    read_dir = arg.read_dir
    save_dir = arg.save_dir
    FACTOR = arg.factor

    img_list = read_images_from_path(read_dir)
    img1, img2 = img_list[0], img_list[1]
    imGray1, imGray2 = map(convert_to_grayscale, (img1, img2))
    (keypoints1, descriptors1), (keypoints2, descriptors2) = map(sift, (imGray1, imGray2))

    matcher = create_matcher(trees=10, checks=50)
    good_matches, points1, points2 = find_good_matches(matcher, keypoints1, descriptors1, keypoints2, descriptors2, factor=FACTOR)

    imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    save_image(save_dir, 'matches.jpg', imMatches)

    K_ = np.asarray(K)
    E, mask = find_essential_matrix(points1, points2, cameraMatrix=K_, method=cv2.RANSAC)
    print(f'Computed Essential matrix: {E}\n')

    points, R, t, new_mask = recover_pose(E, points1, points2, cameraMatrix=K_, mask=mask)
    print(f'Recovered relative Rotation matrix: {R}\n')
    print(f'Recovered relative Translation matrix: {t}\n')

    Matr1 = np.hstack((np.identity(3), np.zeros((3, 1))))
    Matr2 = np.hstack((R, t))

    projMatr1 = K @ Matr1
    projMatr2 = K @ Matr2

    print(f'projMatr1: {projMatr1}\n')
    print(f'projMatr2: {projMatr2}\n')

    points4D = triangulate_points(projMatr1, projMatr2, points1.T, points2.T)
    points3D = cv2.convertPointsFromHomogeneous(points4D.T)
    points3D = np.squeeze(points3D)

    create_ply(points3D, 'ply_files/reconstructed.ply')
    #print('3D Reconstruction completed!')
    create_color_ply(points3D, img1, keypoints1, img2, keypoints2, 'ply_files/reconstructed_color.ply')

    '''
    win_size = 5
    min_disp = -1
    max_disp = 63 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16

    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
                                numDisparities = num_disp,
                                blockSize = 5,
                                uniquenessRatio = 5,
                                speckleWindowSize = 5,
                                speckleRange = 5,
                                disp12MaxDiff = 2,
                                P1 = 8*3*win_size**2,
                                P2 =32*3*win_size**2)

    disparity_map = stereo.compute(img1, img2)

    plt.imshow(disparity_map, 'gray')
    plt.show()

    focal_length = 1698.873755

    Q2 = np.float32([[1,0,0,0],
                [0,-1,0,0],
                [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
                [0,0,0,1]])


    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    colors = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()

    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    print(output_points.shape)

    create_color_ply(output_points, output_colors, 'reconstructed.ply')
    '''
    '''
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    ax.plot(output_points[0], output_points[1], output_points[2], 'b.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=135, azim=90)
    plt.show()
    '''
if __name__ == '__main__':
    main()