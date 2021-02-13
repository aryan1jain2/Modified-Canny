import numpy as np 
import os 
import cv2 
import matplotlib.pyplot as plt 
import argparse
import glob
from math import log10, sqrt

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = Canny_detector(image, lower, upper)
	# return the edged image
	return edged


# defining the canny detector function 

# here weak_th and strong_th are thresholds for 
# double thresholding step 
def Canny_detector(img, weak_th = None, strong_th = None): 
	
	# conversion of image to grayscale 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	cv2.imshow('black',img)

	# Noise reduction step 
	img = cv2.GaussianBlur(img, (5, 5), 1.4) # A 5x5 kernel and sigma value 1.4
	
	# Calculating the gradients 
	gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
	gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 


	# Conversion of Cartesian coordinates to polar 
	mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
	
	# setting the minimum and maximum thresholds 
	# for double thresholding 
	mag_max = np.max(mag) 
	if not weak_th:
		weak_th = mag_max * 0.1
	if not strong_th:
		strong_th = mag_max * 0.5
	
	# getting the dimensions of the input image 
	height, width = img.shape 
	
	# Looping through every pixel of the grayscale 
	# image 
	for i_x in range(width): 
		for i_y in range(height): 
			
			grad_ang = ang[i_y, i_x] 
			grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
			
			# selecting the neighbours of the target pixel 
			# according to the gradient direction 
			# In the x axis direction 
			if grad_ang<= 22.5: 
				neighb_1_x, neighb_1_y = i_x-1, i_y 
				neighb_2_x, neighb_2_y = i_x + 1, i_y 
			
			# top right (diagnol-1) direction 
			elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
				neighb_1_x, neighb_1_y = i_x-1, i_y-1
				neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
			
			# In y-axis direction 
			elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
				neighb_1_x, neighb_1_y = i_x, i_y-1
				neighb_2_x, neighb_2_y = i_x, i_y + 1
			
			# top left (diagnol-2) direction 
			elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
				neighb_1_x, neighb_1_y = i_x-1, i_y + 1
				neighb_2_x, neighb_2_y = i_x + 1, i_y-1
			
			# Now it restarts the cycle 
			elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
				neighb_1_x, neighb_1_y = i_x-1, i_y 
				neighb_2_x, neighb_2_y = i_x + 1, i_y 
			
			# Non-maximum suppression step 
			if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
				if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
					mag[i_y, i_x]= 0
					continue

			if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
				if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
					mag[i_y, i_x]= 0
			 
	ids = np.zeros_like(img) 
	
	# double thresholding step 
	for i_x in range(width): 
		for i_y in range(height): 
			
			grad_mag = mag[i_y, i_x] 
			
			if grad_mag<weak_th: 
				mag[i_y, i_x]= 0
			elif strong_th>grad_mag>= weak_th: 
				ids[i_y, i_x]= 1
			else: 
				ids[i_y, i_x]= 2
				mag[i_y, i_x]= 255
	
	for i in range(width):
		for j in range(height):

			try:
				if(ids[i, j]==1):
					
						if((ids[i+1, j-1] == 2) or (ids[i+1, j] == 2) or (ids[i+1, j+1] == 2)
	        	             or (ids[i, j-1] == 2) or (ids[i, j+1] == 2)
	        	             or (ids[i-1, j-1] == 2) or (ids[i-1, j] == 2) or (ids[i-1, j+1] == 2)):
	
							mag[i, j] = 255
							ids[i, j] = 2
						else:
							mag[i, j] = 0

			except IndexError as e:
				pass
	
	for i in range(width, 0, -1):
		for j in range(height, 0 , -1):

			try:
				if(ids[i, j]==1):
					
						if((ids[i+1, j-1] == 2) or (ids[i+1, j] == 2) or (ids[i+1, j+1] == 2)
	        	             or (ids[i, j-1] == 2) or (ids[i, j+1] == 2)
	        	             or (ids[i-1, j-1] == 2) or (ids[i-1, j] == 2) or (ids[i-1, j+1] == 2)):
	
							mag[i, j] = 255
							ids[i, j] = 2
						else:
							mag[i, j] = 0

			except IndexError as e:
				pass

	for i in range(width): 
		for j in range(height, 0 , -1):

			try:
				if(ids[i, j]==1):
					
						if((ids[i+1, j-1] == 2) or (ids[i+1, j] == 2) or (ids[i+1, j+1] == 2)
	        	             or (ids[i, j-1] == 2) or (ids[i, j+1] == 2)
	        	             or (ids[i-1, j-1] == 2) or (ids[i-1, j] == 2) or (ids[i-1, j+1] == 2)):
	
							mag[i, j] = 255
							ids[i, j] = 2
						else:
							mag[i, j] = 0

			except IndexError as e:
				pass

	for i in range(width, 0, -1):
		for j in range(height):

			try:
				if(ids[i, j]==1):
					
						if((ids[i+1, j-1] == 2) or (ids[i+1, j] == 2) or (ids[i+1, j+1] == 2)
	        	             or (ids[i, j-1] == 2) or (ids[i, j+1] == 2)
	        	             or (ids[i-1, j-1] == 2) or (ids[i-1, j] == 2) or (ids[i-1, j+1] == 2)):
	
							mag[i, j] = 255
							ids[i, j] = 2
						else:
							mag[i, j] = 0

			except IndexError as e:
				pass
	
	# finally returning the magnitude of 
	# gradients of edges 
	return mag 
'''
img = cv2.imread("images/1.jpg")
cv2.imshow('Canny',Canny_detector(img))
cv2.waitKey(0)
'''
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    print('\n')
    print(f"MSE: {mse}")
    print(f"PSNR: {psnr}")

canny_output = r"C:/Users/Apoorva/Desktop/ip_code/images/canny_output"

for i in range (1,10) :
    # Image path 
    image_path = r"C:/Users/Apoorva/Desktop/ip_code/images/"+ f"{i}" + ".jpg"
      # Using cv2.imread() method 
    # to read the image 
    img = cv2.imread(image_path) 
    imgf = auto_canny(img)
      # Change the current directory
    # to specified directory
    os.chdir(canny_output) 
    filename = "savedImage" + f"{i}"+ ".jpg"
    cv2.imwrite(filename, imgf)
