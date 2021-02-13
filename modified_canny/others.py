import cv2
import numpy as np
import os 
from math import log10, sqrt 

def gray_blur(img):
	#converting image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return cv2.GaussianBlur(gray,(3,3),0)


def sobel(img):
	img = gray_blur(img)
	img_sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
	img_sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
	return img_sobelx + img_sobely

def prewitt(img):
	img = gray_blur(img)
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img, -1, kernelx)
	img_prewitty = cv2.filter2D(img, -1, kernely)
	return img_prewittx + img_prewitty

def robert(img):
    img = gray_blur(img)
    kernel_Roberts_x = np.array([[1, 0],[0, -1]])
    kernel_Roberts_y = np.array([[0, -1],[1, 0]])
    img_robertx = cv2.filter2D(img, -1, kernel_Roberts_x)
    img_roberty = cv2.filter2D(img, -1, kernel_Roberts_y)
    return img_robertx + img_roberty

def laplacian(img):
    img = gray_blur(img)
    kernel_laplacian = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    img_laplacian = cv2.filter2D(img, -1, kernel_laplacian)
    return img_laplacian

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0): # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    print(f"MSE:  {mse}") 
    print(f"PSNR: {psnr}\n") 

sobel_output = r"C:/Users/Apoorva/Desktop/ip_code/images/sobel_output"
print('\nSOBEL\n')

for i in range (1,10) :
    # Image path 
    image_path = r"C:/Users/Apoorva/Desktop/ip_code/images/"+ f"{i}" + ".jpg"
      # Using cv2.imread() method 
    # to read the image 
    img = cv2.imread(image_path) 
    imgf = sobel(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    PSNR(img , imgf)
      # Change the current directory
    # to specified directory
    os.chdir(sobel_output) 
    filename = "savedImage" + f"{i}"+ ".jpg"
    cv2.imwrite(filename, imgf)

prewitt_output = r"C:/Users/Apoorva/Desktop/ip_code/images/prewitt_output"
print('\nPREWITT\n')

for i in range (1,10) :
    # Image path 
    image_path = r"C:/Users/Apoorva/Desktop/ip_code/images/"+ f"{i}" + ".jpg"
      # Using cv2.imread() method 
    # to read the image 
    img = cv2.imread(image_path) 
    imgf = prewitt(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    PSNR(img , imgf)
      # Change the current directory
    # to specified directory
    os.chdir(prewitt_output) 
    filename = "savedImage" + f"{i}"+ ".jpg"
    cv2.imwrite(filename, imgf)

cv_canny_output = r"C:/Users/Apoorva/Desktop/ip_code/images/cv_canny_output"
print('\nCV2 CANNY\n') 

for i in range (1,10) :
    # Image path 
    image_path = r"C:/Users/Apoorva/Desktop/ip_code/images/"+ f"{i}" + ".jpg"
      # Using cv2.imread() method 
    # to read the image 
    img = cv2.imread(image_path) 
    imgf = cv2.Canny(img,100,200)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    PSNR(img , imgf)
      # Change the current directory
    # to specified directory
    os.chdir(cv_canny_output) 
    filename = "savedImage" + f"{i}"+ ".jpg"
    cv2.imwrite(filename, imgf)

robert_output = r"C:/Users/Apoorva/Desktop/ip_code/images/robert_output"
print('\nROBERT\n')

for i in range (1,10) :
    # Image path 
    image_path = r"C:/Users/Apoorva/Desktop/ip_code/images/"+ f"{i}" + ".jpg"
      # Using cv2.imread() method 
    # to read the image 
    img = cv2.imread(image_path) 
    imgf = robert(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    PSNR(img , imgf)
      # Change the current directory
    # to specified directory
    os.chdir(robert_output) 
    filename = "savedImage" + f"{i}"+ ".jpg"
    cv2.imwrite(filename, imgf)

laplacian_output = r"C:/Users/Apoorva/Desktop/ip_code/images/laplacian_output"
print('\nLAPLACIAN\n') 

for i in range (1,10) :
    # Image path 
    image_path = r"C:/Users/Apoorva/Desktop/ip_code/images/"+ f"{i}" + ".jpg"
      # Using cv2.imread() method 
    # to read the image 
    img = cv2.imread(image_path) 
    imgf = laplacian(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    PSNR(img , imgf)
      # Change the current directory
    # to specified directory
    os.chdir(laplacian_output) 
    filename = "savedImage" + f"{i}"+ ".jpg"
    cv2.imwrite(filename, imgf)