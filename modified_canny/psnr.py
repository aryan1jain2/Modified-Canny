from math import log10, sqrt 
import cv2 
import numpy as np 

def PSNR(original, compressed): 
	mse = np.mean((original - compressed) ** 2) 
	if(mse == 0): # MSE is zero means no noise is present in the signal . 
				# Therefore PSNR have no importance. 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	print("MSE: " + str(mse))
	print("PSNR: " + str(psnr))

def main(): 
	for j in ["canny_output", "sobel_output", "prewitt_output", "robert_output", "cv_canny_output", "laplacian_output"]:
		print(j)
		for i in range(1,10):
			original = cv2.imread("C:/Users/Apoorva/Desktop/ip_code/images/" + str(i) + ".jpg")
			compressed = cv2.imread("C:/Users/Apoorva/Desktop/ip_code/images/" + j + "/savedImage" + str(i) + ".jpg", 1)
			print("\n" + str(i)) 
			PSNR(original, compressed) 
		print() 



	
if __name__ == "__main__": 
	main() 
