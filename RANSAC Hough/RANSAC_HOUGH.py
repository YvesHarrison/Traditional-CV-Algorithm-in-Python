import numpy as np
import math
import sys
import cv2
import random

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
	conv = np.zeros((pix.shape[0] , pix.shape[1]))
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
			# dx[i][j] = x
			# dy[i][j] = y
			# k = math.sqrt(x ** 2 + y ** 2)
			if sigma == 0:
				conv[i][j] = x
			else:
				conv[i][j] = y
	
	return conv

def Hessian(Ixx, Iyy, Ixy, threshold):

	duplicate = np.zeros((Ixx.shape[0], Ixx.shape[1]))
	for i in range(0, Ixx.shape[0]):
		for j in range(0, Ixx.shape[1]):
			duplicate[i][j] = Ixx[i][j] * Iyy[i][j] - Ixy[i][j] * Ixy[i][j]
	duplicate_1d = duplicate.flatten()
	duplicate_1d = np.sort(duplicate_1d)
	min = duplicate_1d[int(Ixx.shape[0] * Ixx.shape[1] * 0.01)]
	max = duplicate_1d[int(Ixx.shape[0] * Ixx.shape[1] * 0.99)]
	# print(min, max)
	for i in range(0, Ixx.shape[0]):
		for j in range(0, Ixx.shape[1]):
			if duplicate[i][j] < min:
				duplicate[i][j] = 0
			elif duplicate[i][j] > max:
				duplicate[i][j] = 0
			else:
				duplicate[i][j] = int((duplicate[i][j]-min)/(max-min)*255)
			if(duplicate[i][j] < threshold):
				duplicate[i][j] = 0

	return duplicate

def non_maximum_suppression(pix):
	h = pix.shape[0]
	w = pix.shape[1]
	duplicate = np.zeros((h, w))
	for i in range(0, h):
		for j in range(0, w):
			isLocalMax = True
			
			for y in range(i - 1, i + 2):
				for x in range(j - 1, j + 2):
					if x >= 0 and x < w and y >= 0 and y < h:
						if pix[y][x] >= pix[i][j] and (i != y or j != x):
							isLocalMax = False
			if isLocalMax and i > 3 and i < h - 3 and j > 3 and j < w - 3:
				duplicate[i][j] = 255

	return duplicate

def RANSAC(pix, sigma, line_num):
	h = pix.shape[0]
	w = pix.shape[1]
	points = []
	#d = 40
	t = math.sqrt(3.84 * sigma * sigma)
	for i in range(0, h):
		for j in range(0, w):
			if(pix[i][j] == 255):
				points.append(np.array([i, j]))

	lines = [];
	print(len(points))
	random.seed()
	per = 0.95
	num = 0
	while num < 4:
		best_line = 0
		roundline = 0
		best_index = []
		N = sys.maxsize
		while roundline < N:
			roundline += 1
			rand_pos1 = random.randint(0, len(points) - 1)
			rand_pos2 = random.randint(0, len(points) - 1)

			while(rand_pos1 == rand_pos2):
				rand_pos1 = random.randint(0, len(points) - 1)
				rand_pos2 = random.randint(0, len(points) - 1)

			p1 = points[rand_pos1]
			p2 = points[rand_pos2]

			A = p2[0] - p1[0]
			B = p1[1] - p2[1]
			C = p2[1] * p1[0] - p1[1] * p2[0]
			inlier = []

			for i in range(0, len(points)):
				dis = abs(A * points[i][1] + B * points[i][0] + C) / math.sqrt(A * A + B * B)
				if(dis < t):
					inlier.append(i)
			
			if len(inlier) > best_line:
				best_line = len(inlier)
				best_index = inlier
				
			e = 1 - best_line / len(points)
			# print(len(inlier), p1, p2)
			# print(roundline, e, math.log(1 - math.pow((1 - e), 2)))
			N = math.log(1 - per) / math.log(1 - math.pow((1 - e), 2))
		print(len(best_index))
		line = []
		for p in best_index:
			line.append(points[p])
		for p in line:
			pos = -1
			for i in range(0, len(points)):
				if points[i][0] == p[0] and points[i][1] == p[1]:
					pos = i
					break
			if pos != -1:
				points.pop(pos)
		lines.append(line)
		num += 1
		print("find ", num, "th line ",len(points))
	# print(lines)
	return lines

def draw_RANSAC(pix, lines):
	for line in lines:
		min_y = 320
		max_y = 0
		x = 0
		y = 0
		for i in range(0, len(line)):
			if(line[i][0] < min_y):
				min_y = line[i][0]
				x = i
			if(line[i][0] > max_y):
				max_y = line[i][0]
				y = i
		cv2.line(pix, (line[x][1], line[x][0]), (line[y][1], line[y][0]), (255, 0, 0), 1)
	return pix

def Hough(pix, t):
	h = pix.shape[0]
	w = pix.shape[1]
	points = []
	pi = math.pi
	maxr = int(math.ceil(w * math.cos(pi / 4) + math.ceil(h * math.sin(pi / 4))))

	H = np.zeros((maxr + w + 1, 181))
	for i in range(0, h):
		for j in range(0, w):
			if(pix[i][j] == 255):
				points.append(np.array([i, j]))
	print(len(points))
	
	for p in points:
		for theta in range(0, 181):
			xp = p[1] * math.cos(theta * pi / 180)
			yp = p[0] * math.sin(theta * pi / 180)
			rho = int(xp + yp)
			# deal with exceed value
			if rho > maxr:
				continue
			H[rho + w, theta] += 1

	lines = []
	for i in range(0, maxr + w):
		for j in range(0, 181):
			if H[i, j] > t:
				lines.append((i - w, j, H[i, j]))

	return H, lines

def takeThird(elem):
	return elem[2]

def draw_hough(pix, lines):
	lines.sort(key = takeThird, reverse=True)
	pi = math.pi
	for line in lines:
		a = math.cos(line[1] / 180 * pi)
		b = math.sin(line[1] / 180 * pi)
		x = a * line[0]
		y = b * line[0]
		x1 = int(x + 1000 * (-b))
		y1 = int(y + 1000 * (a))
		x2 = int(x - 1000 * (-b))
		y2 = int(y - 1000 * (a))
		cv2.line(pix, (x1, y1), (x2, y2), (255, 0, 0), 1)
	return pix

def main(name):
	image = cv2.imread(name, 0)
	print(image.shape)
	Gaussian_image = cv2.GaussianBlur(image, (3, 3), 0) #Gaussian_Filter(image, 1)
	Ix = Sobel_Filter(Gaussian_image, 0)
	Iy = Sobel_Filter(Gaussian_image, 1)
	Ixy = Sobel_Filter(Ix, 1)
	Iyy = Sobel_Filter(Iy, 1)
	Ixx = Sobel_Filter(Ix, 0)
	# cv2.imwrite("After_Sobel_xx" + name,Ixx)
	# cv2.imwrite("After_Sobel_yy" + name,Iyy)
	# cv2.imwrite("After_Sobel_xy" + name,Ixy)
	Hessian_image = Hessian(Ixx, Iyy, Ixy, 160)
	suppression_image = non_maximum_suppression(Hessian_image)
	image_copy = np.copy(image)
	cv2.imwrite("After_Hessian_" + name,suppression_image)
	ransac = RANSAC(suppression_image, 1, 4)
	ransac_pix = draw_RANSAC(image, ransac)
	cv2.imwrite("After_RANSAC_" + name, ransac_pix)
	hough_pic, hough_line = Hough(suppression_image, 25)
	cv2.imwrite("Hough_" + name, hough_pic)
	hough_pix = draw_hough(image_copy, hough_line)
	cv2.imwrite("After_Hough_" + name, hough_pix)

if __name__=="__main__":
	main(sys.argv[1])






