import os
import cv2
import numpy as np
from PIL import Image
import glob

# Camera Matrix
K = [[1698.873755, 0.000000    , 971.7497705],
     [0.000000   , 1698.8796645, 647.7488275],
     [0.000000   , 0.000000    , 1.000000   ]]

focalX = K[0][0]
focalY = K[1][1]
PP = (K[0][2], K[1][2])

def find_intrinsic():
    rows = 8 # number of inside corners in x
    cols = 6 # number of inside corners in y

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    obj_points_arr = []
    img_points_arr = []

    # Make a list of calibration images
    save_resized_images('checkerboard', 'checkerboard_resized/checkerboard_resized')
    images = glob.glob('checkerboard_resized/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            obj_points_arr.append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            img_points_arr.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

        cv2.imshow('checker board', img)
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_arr, img_points_arr, gray.shape[::-1], None, None)
    print(f'mtx: {mtx}')
    
    return mtx
    '''
    error = 0

    for i in range(len(obj_points_arr)):
        img_points, _ = cv2.projectPoints(obj_points_arr[i], rvecs[i], tvecs[i], mtx, dist)
        error += cv2.norm(img_points_arr[i], img_points, cv2.NORM_L2) / len(img_points)

    print(f'Total error: {error / len(obj_points_arr)}')

    # Load one of the test images
    img = cv2.imread('checkerboard_resized/checkerboard_resized1.jpg')
    h, w = img.shape[:2]

    # Obtain the new camera matrix and undistort the image
    newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print(f'newCameraMtx: {newCameraMtx}')
    undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)

    # Crop the undistorted image
    # x, y, w, h = roi
    # undistortedImg = undistortedImg[y:y + h, x:x + w]

    # Display the final result
    cv2.imshow('chess board', np.hstack((img, undistortedImg)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

def create_ply(points, filename):
    ply_header = '''ply
        format ascii 1.0
        element vertex %(point_num)d
        property float x
        property float y
        property float z
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(point_num=len(points)))
        np.savetxt(f, points, '%f %f %f')


def create_color_ply(points, image1, keypoints1, image2, keypoints2, filename):
    with open(filename, 'w') as f:

        def writeline(f, line):
            return f.write("{}\n".format(line))

        writeline(f, 'ply')
        writeline(f, 'format ascii 1.0')
        writeline(f, 'element vertex {}'.format(len(points)))
        writeline(f, 'property float x')
        writeline(f, 'property float y')
        writeline(f, 'property float z')
        writeline(f, 'property uchar red')
        writeline(f, 'property uchar green')
        writeline(f, 'property uchar blue')
        writeline(f, 'end_header')

        def get_keypoint_color(img, keypoints, index):
            red = []
            green = []
            blue = []

            xcoord = int(keypoints[index].pt[0])
            ycoord = int(keypoints[index].pt[1])

            for y in range(ycoord - 3, ycoord + 3):
                for x in range(xcoord - 3, xcoord + 3):
                    if 0 <= y < len(img) and 0 <= x < len(img[0]):
                        red.append(img[y][x][2])
                        green.append(img[y][x][1])
                        blue.append(img[y][x][0])

            return (sum(red) / len(red), sum(green) / len(green), sum(blue) / len(blue))

        for row_num in range(len(points)):
            row = points[row_num]
            line = '%f %f %f' % (row[0], row[1], row[2])

            color1 = get_keypoint_color(image1, keypoints1, row_num)
            color2 = get_keypoint_color(image2, keypoints2, row_num)
            red = (color1[0] + color2[0]) / 2
            green = (color1[1] + color2[1]) / 2
            blue = (color1[2] + color2[2]) / 2
            line2 = '%i %i %i' % (red, green, blue)

            writeline(f, line + ' ' + line2)

def resize_image(img_path, width=640, height=640):
    img = Image.open(img_path)
    resize_image = img.resize((width, height))
    return resize_image

def save_resized_image(img, save_path, width=640, height=640):
    resize_image(img, width, height).save(save_path, 'JPEG', quality=95)

def save_resized_images(img_path, save_path, width=640, height=640):
    images = glob.glob(os.path.join(img_path, '*.jpg'))
    for i, image in enumerate(images):
        save_resized_image(image, save_path+str(i+1)+'.jpg', width, height)
        #save_image(save_path, save_path + str(i+1), resize_image(image, width, height))

def convert_to_grayscale(img):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imGray

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img

def save_image(save_path, save_as, img):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    cv2.imwrite(save_path + save_as, img)

def read_images_from_path(img_path):
    img_list = []
    
    for root, dirs, files in os.walk(img_path):
        for file in files:
            img_list.append(read_image(root + file))

    return img_list