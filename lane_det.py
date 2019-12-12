import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math

from imp import reload #in 2.7 reload is in imp not importlib
# import utils; reload(utils)
# from utils import *

calibration_dir = "/home/ajwahir/farws/laneDet/CarND-Advanced-Lane-Lines/camera_cal"
test_imgs_dir = "/home/ajwahir/farws/laneDet/CarND-Advanced-Lane-Lines/test_images"
output_imgs_dir = "/home/ajwahir/farws/laneDet/CarND-Advanced-Lane-Lines/output_images"
output_videos_dir = "/home/ajwahir/farws/laneDet/CarND-Advanced-Lane-Lines/output_videos"

# Let's get all our calibration image paths
cal_imgs_paths = glob.glob(calibration_dir + "/*.jpg") #array into which the 

# Let's the first chessboard image to see what it looks like
cal_img_path = cal_imgs_paths[11]
# cal_img = load_image(cal_img_path) load_image is a matplolib func so replace it with cv2.imread which stores the image in an array
cal_img = cv2.imread(cal_img_path)
# cv2.imshow("checker board",cal_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


cx = 9
cy = 6

def to_grayscale(mat):
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    return gray


def findChessboardCorners(img, nx, ny):
    """
    Finds the chessboard corners of the supplied image (must be grayscale)
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    return cv2.findChessboardCorners(img, (nx, ny), None)

def showChessboardCorners(img, nx, ny, ret, corners):
    """
    Draws the chessboard corners of a given image
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    ret and corners should represent the results from cv2.findChessboardCorners()
    """
    c_img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    #plt.axis('off')
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

ret, corners = findChessboardCorners(to_grayscale(cal_img), cx, cy)
showChessboardCorners(cal_img, cx, cy, ret, corners) # ret is the variable in which the distortion matrix is stored

def findImgObjPoints(imgs_paths, nx, ny):
    """
    Returns the objects and image points computed for a set of chessboard pictures taken from the same camera
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    objpts = []
    imgpts = []
    
    # Pre-compute what our object points in the real world should be (the z dimension is 0 as we assume a flat surface)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    for img_path in imgs_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# cv2.COLOR_BGR2GRAY is just converting the RGB image to grayscale
        ret, corners = findChessboardCorners(gray, nx, ny)
        
        if ret:
            # Found the corners of an image
            imgpts.append(corners)
            # Add the same object point since they don't change in the real world
            objpts.append(objp)
    
    return objpts, imgpts

opts, ipts = findImgObjPoints(cal_imgs_paths, cx, cy)

def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, to_grayscale(img).shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

cal_img_example = cv2.imread(cal_imgs_paths[0])
cal_img_undist = undistort_image(cal_img_example, opts, ipts)
# cv2.imshow("example_image",cal_img_example)
# cv2.imshow("undistorted_img",cal_img_undist)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# fig, ax = plt.subplots(1, 2, figsize=(10,7))
# ax[0].imshow(cal_img_example)
# ax[0].axis("off")
# ax[0].set_title("Distorted Image")

# ax[1].imshow(cal_img_undist)
# ax[1].axis("off")
# ax[1].set_title("Undistorted Image")

# cv2.imshow()

test_img_path = "/home/ajwahir/farws/laneDet/CarND-Advanced-Lane-Lines/test_images/straight_lines2.jpg" # this is the test image to be converted to hls, hls is better since it is better to detect colors
undistorted_test_img = cv2.imread(test_img_path)



def compute_hls_white_yellow_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    hls_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2HLS)
    # print hls_img
    
    # Compute a binary thresholded image where yellow is isolated from HLS components
    img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                 & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                 & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                ] = 255
    # print img_hls_yellow_bin
    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hls_img[:,:,0]) # here predefined values in B G and R channels are given to classify the yellow color
    img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                 & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                 & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                ] = 255 
    
    # Now combine both
    img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 255) | (img_hls_white_bin == 255)] = 255 #or statement to select the hue if it is yellow or white depending on the conditions above
    # print img_hls_white_yellow_bin

    return img_hls_white_yellow_bin

undistorted_yellow_white_hls_img_bin = compute_hls_white_yellow_binary(undistorted_test_img)

# cv2.imshow("undistorted_yellow_white_hls_img_bin",undistorted_yellow_white_hls_img_bin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def abs_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) 
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs)) # scale the magnitude of gradient as if a pixel value is 255*sqrt(2) it would exceed the threshold limit
    
    gradient_mask = np.zeros_like(sobel_scaled) # mask creates a copy matrix with zeros and after thresholding replaces these values in the mask matrix
    gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 255 #binary thresholding 
    return gradient_mask




sobx_best = abs_sobel(undistorted_yellow_white_hls_img_bin, kernel_size=15, thres=(20, 120))
soby_best = abs_sobel(undistorted_yellow_white_hls_img_bin, x_dir=False, kernel_size=15, thres=(20, 120))

def mag_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled_x = np.uint8(255 * sobel / np.max(sobel_abs))

    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled_y = np.uint8(255 * sobel / np.max(sobel_abs))

    sxy_mag = np.sqrt(np.square(sobel_scaled_x) + np.square(sobel_scaled_y))
    sxy_mag = np.uint8(255 * sxy_mag / np.max(sxy_mag))
    
    gradient_mask = np.zeros_like(sxy_mag)
    gradient_mask[(thres[0] <= sxy_mag) & (sxy_mag <= thres[1])] = 255 #binary thresholding 
    
    return gradient_mask

sob_mag_best = mag_sobel(undistorted_yellow_white_hls_img_bin, kernel_size=15, thres=(80, 200))

def direction_threshold(gray_img,kernel_size=3,angle_thres=(0,np.pi/2)):
	sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
	sobel_abs_x = np.absolute(sobel)
	# sobel_scaled_x = np.uint8(255 * sobel / np.max(sobel_abs))

	sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
	sobel_abs_y = np.absolute(sobel)
	# sobel_scaled_y = np.uint8(255 * sobel / np.max(sobel_abs))

	dir_sxy = np.arctan2(sobel_abs_x,sobel_abs_y)

	direction_mask = np.zeros_like(dir_sxy)
	direction_mask[(angle_thres[0]<=dir_sxy) & (dir_sxy<=angle_thres[1])] = 255
	return direction_mask


dir_sobel = direction_threshold(undistorted_yellow_white_hls_img_bin, 15,(np.pi/4,np.pi/2))


combined = np.zeros_like(dir_sobel)
combined[(sobx_best == 255) | ((soby_best == 255) & (sob_mag_best == 255) & (dir_sobel == 255))] = 255


# cv2.imshow("direction",combined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

color_binary = np.dstack((np.zeros_like(combined), combined, undistorted_yellow_white_hls_img_bin)) * 255
# color_binary = color_binary.astype(np.uint8)

combined_binary = np.zeros_like(undistorted_yellow_white_hls_img_bin)
combined_binary[(combined == 1) | (undistorted_yellow_white_hls_img_bin == 1)] = 1

combined_binaries = [[color_binary, combined_binary]]
combined_binaries_lbs = np.asarray([["Stacked Thresholds", "Combined Color And Gradient Thresholds"]])

# show_image_list(combined_binaries, combined_binaries_lbs, "Color And Binary Combined Gradient And HLS (S) Thresholss", cols=2, fig_size=(17, 6), show_ticks=False)

copy_combined = np.copy(undistorted_test_img)
(bottom_px, right_px) = (copy_combined.shape[0] - 1, copy_combined.shape[1] - 1) 
pts = np.array([[210,bottom_px],[595,450],[690,450], [1110, bottom_px]], np.int32)
cv2.polylines(copy_combined,[pts],True,(255,0,0), 10)
# plt.axis('off')
# plt.imshow(copy_combined)

# cv2.imshow("perspective_trans",copy_combined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def compute_perspective_transform_matrices(src, dst):
    """
    Returns the tuple (M, M_inv) where M represents the matrix to use for perspective transform
    and M_inv is the matrix used to revert the transformed image back to the original one
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    return (M, M_inv)

def perspective_transform(img, src, dst):   
    """
    Applies a perspective 
    """
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

src_pts = pts.astype(np.float32)
dst_pts = np.array([[200, bottom_px], [200, 0], [1000, 0], [1000, bottom_px]], np.float32)

test_img_persp_tr = perspective_transform(combined, src_pts, dst_pts)
test_img_Color_persp_tr = perspective_transform(undistorted_test_img, src_pts, dst_pts)
test_img_persp_tr= test_img_persp_tr.astype(np.uint8)
kernel=np.ones((11,11),np.uint8)
opening = cv2.morphologyEx(test_img_persp_tr,cv2.MORPH_OPEN,kernel)
edges = cv2.Canny(test_img_persp_tr,100,200)

lines = cv2.HoughLines(edges, 2, np.pi / 180, 150, None, 0, 0)
# lines = cv2.HoughLinesP(edges,2,np.pi/180,150,None,3,2)
# print len(lines)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(test_img_Color_persp_tr, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# cv2.imshow("persTransformed",edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.imshow("persTransformed",test_img_persp_tr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



cv2.imshow("persTransformed",test_img_Color_persp_tr)
cv2.waitKey(0)
cv2.destroyAllWindows()





























