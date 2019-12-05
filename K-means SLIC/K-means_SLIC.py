import numpy as np
import math
import sys
import cv2
import random
import time 

def K_means(image, k):
	height = image.shape[0]
	width = image.shape[1]
	center = []
	for i in range(0 ,k):
		x = random.randint(0, 255)
		y = random.randint(0, 255)
		z = random.randint(0, 255)
		center.append([x, y, z])
	# print(center)
	table = np.zeros((height , width))
	res = image.copy()
	
	(image_0, image_1, image_2) = cv2.split(image)
	image_0 = np.array(image_0, dtype = int)
	image_1 = np.array(image_1, dtype = int)
	image_2 = np.array(image_2, dtype = int)
	print("Start K-means")
	error_rate = 1
	while error_rate > 0.01:
		tmp = []
		layer = []
		error = 0
		n_t = time.process_time()
		for i in range(0, k):
			tmp.append([])
			layer.append([])

		for i in range(0, k):
			min0 = image_0 - center[i][0] 
			min1 = image_1 - center[i][1]
			min2 = image_2 - center[i][2]
			layer[i] = np.multiply(min0, min0) + np.multiply(min1, min1) + np.multiply(min2, min2)
		
		for i in range(0, height):
			for j in range(0, width):
				max_dis = 225000
				loc = -1
				for n in range(0, k):
					if layer[n][i][j] < max_dis:
						loc = n
						max_dis = layer[n][i][j]
				tmp[loc].append([image[i][j][0], image[i][j][1], image[i][j][2]])
				if table[i][j] != loc:
					error = error + 1
				table[i][j] = loc

		for i in range(0, k):
			if len(tmp[i])!= 0:
				s = np.sum(tmp[i], axis = 0)
				sum_x = s[0]
				sum_y = s[1]
				sum_z = s[2]
				center[i] = [int(sum_x / len(tmp[i])), int(sum_y / len(tmp[i])), int(sum_z / len(tmp[i]))]
			else:
				x = random.randint(0, 255)
				y = random.randint(0, 255)
				z = random.randint(0, 255)
				center[i] = [x, y, z]
		n_c = time.process_time()
		print("time", 1000 * (n_c - n_t),"ms")
		error_rate = error / (width * height)
		print(error, error_rate)

	for i in range(0, height):
		for j in range(0, width):
			res[i][j][0] = center[int(table[i][j])][0]
			res[i][j][1] = center[int(table[i][j])][1]
			res[i][j][2] = center[int(table[i][j])][2]
	print("K-means finished")
	return res

def SLIC(image, size, max_iter):
	image_pile = []
	center = []
	image = np.array(image, dtype = int)
	#Divide the image in blocks of 50*50 pixels 
	for i in range(0, int(image.shape[0] / size)):
		for j in range(0, int(image.shape[1] / size)):
			image_pile.append(image[i * size : (i + 1) * size, j * size : (j + 1) * size])
			center.append([int(i * size + size / 2), int(j * size + size / 2)])
	
	cn = 0
	(image_0, image_1, image_2) = cv2.split(image)
	image_0 = np.array(image_0, dtype = int)
	image_1 = np.array(image_1, dtype = int)
	image_2 = np.array(image_2, dtype = int)
	local_i = np.zeros((image.shape[0], image.shape[1]))
	local_j = np.zeros((image.shape[0], image.shape[1]))
	for i in range(0, image.shape[0]):
		for j in range(0, image.shape[1]):
			local_i[i][j] = i
			local_j[i][j] = j
	print("SLIC")
	zero = np.uint8(0)
	# print(center)
	scale = int(math.sqrt(2) * 50) + 1
	max_d = 256 ** 2 * 3 + image.shape[0] ** 2 + image.shape[1] ** 2 

	while cn < max_iter:
		#local shift
		cn = cn + 1
		print("Round", cn)
		for k in range(0, len(center)):
			max_gra = 5120
			tmp = center[k]
			for i in range(center[k][0] - 1, center[k][0] + 2):
				for j in range(center[k][1] - 1, center[k][1] + 2):
					x0 = (image[i][j + 1][0] + 2 * image[i + 1][j + 1][0] + image[i + 2][j + 1][0]) - (image[i][j][0] + 2 * image[i + 1][j][0] + image[i + 2][j][0])
					y0 = (image[i][j][0] + 2 * image[i][j + 1][0] + image[i][j + 2][0]) - (image[i + 2][j][0] + 2 * image[i + 2][j + 1][0] + image[i + 2][j + 2][0])
					x1 = (image[i][j + 1][1] + 2 * image[i + 1][j + 1][1] + image[i + 2][j + 1][1]) - (image[i][j][1] + 2 * image[i + 1][j][1] + image[i + 2][j][1])
					y1 = (image[i][j][1] + 2 * image[i][j + 1][1] + image[i][j + 2][1]) - (image[i + 2][j][1] + 2 * image[i + 2][j + 1][1] + image[i + 2][j + 2][1])
					x2 = (image[i][j + 1][2] + 2 * image[i + 1][j + 1][2] + image[i + 2][j + 1][2]) - (image[i][j][2] + 2 * image[i + 1][j][2] + image[i + 2][j][2])
					y2 = (image[i][j][2] + 2 * image[i][j + 1][2] + image[i][j + 2][2]) - (image[i + 2][j][2] + 2 * image[i + 2][j + 1][2] + image[i + 2][j + 2][2])
					gra = math.sqrt(x0 ** 2 + y0 ** 2 + x1 ** 2 + y1 ** 2 + x2 ** 2 + y2 ** 2)
					if gra < max_gra:
						tmp = [i, j]
						max_gra = gra
			center[k] = tmp
				
		tmp = []
		layer = []
		for i in range(0, len(center)):
			tmp.append([])
			layer.append([])
		table = np.zeros((image.shape[0], image.shape[1]))
		# calculate distance on all channels
		for i in range(0, len(center)):
			min0 = image_0 - image[center[i][0], center[i][1]][0] 
			min1 = image_1 - image[center[i][0], center[i][1]][1]
			min2 = image_2 - image[center[i][0], center[i][1]][2]
			minx = (local_i - center[i][0]) * 2
			miny = (local_j - center[i][1]) * 2
			layer[i] = np.multiply(min0, min0) + np.multiply(min1, min1) + np.multiply(min2, min2) +  np.multiply(minx, minx) +  np.multiply(miny, miny)
		
		for i in range(0, image.shape[0]):
			for j in range(0, image.shape[1]):
				max_dis = max_d
				loc = -1
				for k in range(0, len(center)):
					if(i <= center[k][0] + scale and i >= center[k][0] - scale and j <= center[k][1] + scale and j >= center[k][1] - scale):
						# dis = (image_0[i][j] - image[center[k][0]][center[k][1]][0]) ** 2 + (image_1[i][j] - image[center[k][0]][center[k][1]][1]) ** 2 + (image_2[i][j] - image[center[k][0]][center[k][1]][2]) ** 2 + ((i - center[k][0]) / 2) ** 2 + ((j - center[k][1]) / 2) ** 2
						if layer[k][i][j] < max_dis:
							max_dis = layer[k][i][j]
							loc = k
				if loc != -1:
					table[i][j] = loc
					tmp[loc].append([i, j])
		#update center
		for i in range(0, len(center)):
			if len(tmp[i])!= 0:
				s = np.sum(tmp[i], axis = 0)
				sum_x = s[0]
				sum_y = s[1]
				center[i] = [int(sum_x / len(tmp[i])), int(sum_y / len(tmp[i]))]
		
	res = image.copy()
	res_line = image.copy()
	for i in range(0, image.shape[0]):
		for j in range(0, image.shape[1]):
			x = center[int(table[i][j])][0]
			y = center[int(table[i][j])][1]
			res[i][j][0] = image[x][y][0]
			res[i][j][1] = image[x][y][1]
			res[i][j][2] = image[x][y][2]
			if check(i,j,table,table[i][j]):
				res_line[i][j][0] = image[x][y][0]
				res_line[i][j][1] = image[x][y][1]
				res_line[i][j][2] = image[x][y][2]
			else:
				res_line[i][j][0] = zero
				res_line[i][j][1] = zero
				res_line[i][j][2] = zero
	
	print("SLIC finished")
	return res, res_line

def check(x,y,table,cluster):
	for i in range(x - 1, x + 2):
		for j in range(y - 1, y + 2):
			if i >= 0 and j >= 0 and i < len(table) and j < len(table[0]):
				if table[i][j] != cluster:
					return False
	return True

def main(name1, name2):
	image = cv2.imread(name1)
	k_image = K_means(image, 10)
	cv2.imwrite("After_Kmeans_" + name1, k_image)
	image1 = cv2.imread(name2)
	SLIC_image, SLIC_line = SLIC(image1, 50, 3)
	cv2.imwrite("After_SLIC_No_Line_" + name2, SLIC_image)
	cv2.imwrite("After_SLIC_Line_" + name2, SLIC_line)

if __name__=="__main__":
	main(sys.argv[1], sys.argv[2])
