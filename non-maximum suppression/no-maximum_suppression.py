from PIL import Image
import numpy as np
import math
import sys
#read in pgm file and transform it into numpy
def read_image(name):
    image = Image.open(name)
    pix = np.array(image)
    return pix, image.size[0], image.size[1], image

#generate 2D Gaussian kernel
def gaussian_2d_kernel(kernel_size, sigma):
	kernel_x = np.zeros((1, kernel_size))
	kernel_y = np.zeros((1, kernel_size))
	center = kernel_size // 2

	if sigma == 0:
		sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
	constant = 2 * (sigma ** 2)
	sum_val = 0;

	#generate 1D kernel from x, y direction
	for i in range(0, kernel_size):
		x = i - center
		kernel_x[0][i] = (1 / (sigma * math.sqrt(2 * math.pi)))* np.exp(-(x ** 2) / constant)
		kernel_y[0][i] = (1 / (sigma * math.sqrt(2 * math.pi)))* np.exp(-(x ** 2) / constant)
		sum_val += kernel_x[0][i]
	# ensure sum of kernel be 1
	sum_val = 1 / sum_val
	kernel_x = kernel_x * sum_val
	kernel_y = kernel_y * sum_val

	# generate 2D kenerl using matrix multiplication
	kernel_2d = np.multiply(kernel_x.T, kernel_y)
	return kernel_2d

def sum2D(img, fliter):
	res = (img * fliter).sum()
	if(res > 255):
		return 255
	else:
		return res

def Gaussian_Filter(pix, sigma):
	kernel = gaussian_2d_kernel(6 * sigma + 1, sigma)

	k_heigh = kernel.shape[0]
	k_width = kernel.shape[1]
	d_heigh = pix.shape[0] + k_heigh - 1
	d_width = pix.shape[1] + k_width - 1
	duplicate = np.zeros((d_heigh, d_width), dtype = 'uint8')

	conv = np.zeros((pix.shape[0] , pix.shape[1]), dtype = 'uint8')
	size_h = kernel.shape[0] // 2
	size_w = kernel.shape[1] // 2
	# generate picture with zero convolution edges to ensure the picture after fliter has the same size
	for i in range(0, pix.shape[0]):
		for j in range(0, pix.shape[1]):
			duplicate[i + size_h][j + size_w] = pix[i][j]
	# calculate convolution
	for i in range(0, pix.shape[0]):
		for j in range(0, pix.shape[1]):
			conv[i][j] = sum2D(duplicate[i:i + k_heigh, j:j + k_width], kernel)
	return conv

#Gradient calculation
def Sobel_Filter(pix, sigma):
	duplicate = np.zeros((pix.shape[0] + 2, pix.shape[1] + 2))
	conv = np.zeros((pix.shape[0] , pix.shape[1]), dtype = 'uint8')
	dx = np.zeros((pix.shape[0] , pix.shape[1]))
	dy = np.zeros((pix.shape[0] , pix.shape[1]))
	for i in range(0, pix.shape[0]):
		for j in range(0, pix.shape[1]):
			duplicate[i + 1][j + 1] = pix[i][j]

	# calculate convolution
	for i in range(0, pix.shape[0]):
		for j in range(0, pix.shape[1]):
			x = (duplicate[i][j + 1] + 2 * duplicate[i + 1][j + 1] + duplicate[i + 2][j + 1]) - (duplicate[i][j] + 2 * duplicate[i + 1][j] + duplicate[i + 2][j])
			y = (duplicate[i][j] + 2 * duplicate[i][j + 1] + duplicate[i][j + 2]) - (duplicate[i + 2][j] + 2 * duplicate[i + 2][j + 1] + duplicate[i + 2][j + 2])
			dx[i][j] = x
			dy[i][j] = y
			k = math.sqrt(x ** 2 + y ** 2)
			if(k > 255 / (4 * sigma)):
				k = 255
			else:
				k = 0
			conv[i][j] = k
	
    # find threshold 
	# max1 = 0
	# for i in range (0, pix.shape[0]):
	# 	for j in range (0, pix.shape[1]):
	# 		if(conv[i][j] > max1):
	# 			max1 = conv[i][j]
    
	# k = int(max1 / (4* sigma))

	# for i in range(0, pix.shape[0]):
	# 	for j in range(0, pix.shape[1]):
	# 		if conv[i][j] < k:
	# 			conv[i][j] = 0
	# 		else:
	# 			conv[i][j]=255
	return conv, dx, dy

def save_image(name, pix):
	image = Image.fromarray(pix)
	image.save(name)

def non_maximum_suppression(pix, dx, dy):
	duplicate = np.zeros((pix.shape[0], pix.shape[1]), dtype = 'uint8')
	P1 = math.pi / 6
	P2 = math.pi / 3
	for i in range(1, pix.shape[0] - 1):
		for j in range(1, pix.shape[1] - 1):
			if(pix[i][j] != 0):
				#vertical
				if(dx[i][j] == 0 and dy[i][j] != 0):
					if(pix[i][j] > pix[i + 1][j] or pix[i][j] > pix[i - 1][j]):
						duplicate[i][j] = pix[i][j]
					else:
						duplicate[i][j] = 0
				#horizontal
				elif(dx[i][j] != 0 and dy[i][j] == 0):
					if(pix[i][j] > pix[i][j + 1] or pix[i][j] > pix[i][j - 1]):
						duplicate[i][j] = pix[i][j]
					else:
						duplicate[i][j] = 0
				elif(dx[i][j] != 0 and dy[i][j] != 0):
					angle = math.atan(dy[i][j] / dx[i][j])
					#horizontal
					if(angle <= P1 and angle >= -P1):
						if(pix[i][j] > pix[i][j + 1] or pix[i][j] > pix[i][j - 1]):
							duplicate[i][j] = pix[i][j]
						else:
							duplicate[i][j] = 0
					#diagonal
					elif(angle > P1 and angle <= P2):
						if(pix[i][j] > pix[i + 1][j + 1] or pix[i][j] > pix[i - 1][j - 1]):
							duplicate[i][j] = pix[i][j]
						else:
							duplicate[i][j] = 0
					# diagonal
					elif(angle < -P1 and angle >= -P2):
						if(pix[i][j] > pix[i + 1][j - 1] or pix[i][j] > pix[i - 1][j + 1]):
							duplicate[i][j] = pix[i][j]
						else:
							duplicate[i][j] = 0
					#vertical
					elif(angle > P2 or angle < -P2):
						if(pix[i][j] > pix[i + 1][j] or pix[i][j] > pix[i - 1][j]):
							duplicate[i][j] = pix[i][j]
						else:
							duplicate[i][j] = 0
	return duplicate

def Canny(name, sigma):
	pix, width, height, image = read_image(name)
	name = name[:len(name) - 4]
	save_image(name + '_original_sigma = '+ str(sigma) + '.jpg', pix)
	filtered = Gaussian_Filter(pix, sigma)
	save_image(name + '_gaussian_sigma = '+ str(sigma) + '.jpg', filtered)
	gradient, dx, dy = Sobel_Filter(filtered, sigma)
	save_image(name + '_gradient_sigma = '+ str(sigma) + '.jpg', gradient)
	suppression = non_maximum_suppression(gradient, dx, dy)
	save_image(name + '_canny_sigma = '+ str(sigma) + '.jpg', suppression)
	return name

def main(argv):
	if(len(argv) != 3):
		print("Two arguments needed, get more or less!")
	else:
		Canny(argv[1], int(argv[2]))

if __name__=="__main__":
	main(sys.argv)
# Canny("kangaroo.pgm", 1)
# Canny("plane.pgm", 1)
# Canny("red.pgm", 1)
# Canny("kangaroo.pgm", 2)
# Canny("plane.pgm", 2)
# Canny("red.pgm", 2)
# Canny("kangaroo.pgm", 3)
# Canny("plane.pgm", 3)
# Canny("red.pgm", 3)