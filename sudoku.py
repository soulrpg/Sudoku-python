import cv2
from skimage import data, io, filters, morphology, feature
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib import colors
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import erosion, dilation, disk
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks
import os
import math
from tensorflow.keras.models import load_model
 
def getPrediction(image, model):
    result = []
    ## Preparing img so it can be used by model
    #img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(image.copy())
    cutx = int(0.15 * img.shape[1])
    cuty = int(0.15 * img.shape[0])
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if (x < cutx or y < cuty):
                img[y][x] = 255
                img[-y][x] = 255
                img[y][-x] = 255
                img[-y][-x] = 255
    #img = img[cuty:img.shape[0] - cuty, cutx:img.shape[1] - cutx]
    #cv2.imshow('Whited', img)
    img2 = cv2.resize(img, (28, 28))
    #cv2.imshow('Prepared', img2)
 
    img2 = img2.reshape(1, 28, 28, 1)
    ## Using model
    predictions = model.predict(img2)
    classIndex = model.predict_classes(img2)
    probabilityValue = np.amax(predictions)
    result.append(predictions)
    result.append([classIndex, probabilityValue])
    print(result)
    return result
 
 
# Checks if two lines [[x1, y1], [x2, y2]] intersect. Returns a point [x, y] or None
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
 
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
 
    div = det(xdiff, ydiff)
    if div == 0:
        return None
 
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]
 
 
# Returns True if there is number in cell or False if there is not (needs to be upgraded)
def number_inside(cell, percent):
    h, w = cell.shape
    searching_x = w // 3
    searching_y = h // 3
    middle = cell[searching_y // 2:int(searching_y * 1.5), searching_x // 2:int(searching_x * 1.5)]
    # Count white pixels in the middle
    white_pixels = 0
    print("Middle")
    print("Total number of pixels in the middle:", searching_x * searching_y)
    for i in range(0, len(middle)):
        for j in range(0, len(middle[i])):
            if middle[i][j] == 255:
                white_pixels += 1
    print("Percent of white pixels:", white_pixels / (searching_x * searching_y))
    return white_pixels / (searching_x * searching_y) > percent
 
 
def process(image, model):
    sudoku_pic = image
    blurred = cv2.GaussianBlur(sudoku_pic, (13, 13), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 5, 2)
    binary = cv2.bitwise_not(binary)
    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    # binary = cv2.dilate(binary, kernel)
 
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 720)
    h, theta, d = hough_line(binary, theta=tested_angles)
    origin = np.array((0, binary.shape[1]))
    ax = plt.gca()
    ax.imshow(binary, cmap="gray")
 
    lines = []
    intersect_points = []
 
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=np.max(h) * 0.7, min_distance=50)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax.plot(origin, (y0, y1), '-r')
        lines.append([[float(origin[0]), y0], [float(origin[1]), y1]])
    print("Shape:", image.shape)
    for line in lines:
        for line2 in lines:
            if not (line == line2):
                res = line_intersection(line, line2)
                if not (res == None):
                    intersect_points.append(res)
 
    top_left = [image.shape[1], image.shape[0]]
    top_right = [0, image.shape[0]]
    bottom_left = [image.shape[1], 0]
    bottom_right = [0, 0]
    for point in intersect_points:
        if point[0] >= 0 and point[0] <= image.shape[1] and point[1] >= 0 and point[1] <= image.shape[0]:
            # top_left
            diff_x = point[0] - top_left[0] #should be negative
            diff_y = point[1] - top_left[1] #should be negative
            if point[0] < top_left[0] and point[1] < top_left[1]:
                top_left = point
            elif point[0] < top_left[0] and abs(diff_x) > abs(diff_y) * 2:
                top_left = point
            elif point[1] < top_left[1] and abs(diff_y) > abs(diff_x) * 2:
                top_left = point
            # top_right
            diff_x = point[0] - top_right[0]
            diff_y = point[1] - top_right[1]
            if point[0] > top_right[0] and point[1] < top_right[1]:
                top_right = point
            elif point[0] > top_right[0] and abs(diff_x) > abs(diff_y) * 2:
                top_right = point
            elif point[1] < top_right[1] and abs(diff_y) > abs(diff_x) * 2:
                top_right = point
            # bottom_left
            diff_x = point[0] - bottom_left[0]
            diff_y = point[1] - bottom_left[1]
            if point[0] < bottom_left[0] and point[1] > bottom_left[1]:
                bottom_left = point
            elif point[0] < bottom_left[0] and abs(diff_x) > abs(diff_y) * 2:
                bottom_left = point
            elif point[1] > bottom_left[1] and abs(diff_y) > abs(diff_x) * 2:
                bottom_left = point
            # bottom_right
            diff_x = point[0] - bottom_right[0]
            diff_y = point[1] - bottom_right[1]
            if point[0] > bottom_right[0] and point[1] > bottom_right[1]:
                bottom_right = point
            elif point[0] > bottom_right[0] and abs(diff_x) > abs(diff_y) * 2:
                bottom_right = point
            elif point[1] > bottom_right[1] and abs(diff_y) > abs(diff_x) * 2:
                bottom_right = point
 
    #ax.scatter(min_x, min_y, s=25)
    ax.plot(top_left[0], top_left[1], 'bo')
    ax.plot(top_right[0], top_right[1], 'bo')
    ax.plot(bottom_left[0], bottom_left[1], 'bo')
    ax.plot(bottom_right[0], bottom_right[1], 'bo')
    ax.set_xlim(origin)
    ax.set_ylim((binary.shape[0], 0))
    ax.set_axis_off()
    ax.set_title('Sudoku')
    plt.tight_layout()
 
    plt.show()
    plt.close()
    
    ################################
    pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
    pts2 = np.float32([[0, 0],[900, 0],[900, 900],[0, 900]])
    
      
    # Apply Perspective Transform Algorithm 
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    projected_img = cv2.warpPerspective(sudoku_pic, matrix, (900, 900))
    
    cv2.imshow("Projected", projected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    # Cropping out cells from grid
    x_diff = projected_img.shape[1] // 9
    y_diff = projected_img.shape[0] // 9
    cells = []
    for i in range(9):
        for j in range(9):
            lower_bound_y = i * y_diff
            upper_bound_y = (i + 1) * y_diff
            lower_bound_x = j * x_diff
            upper_bound_x = (j + 1) * x_diff
            cell = projected_img[lower_bound_y:upper_bound_y, lower_bound_x:upper_bound_x]
            cell = cv2.GaussianBlur(cell, (3, 3), 0)
            cell = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 5, 2)
            cell = cv2.bitwise_not(cell)
            cells.append(cell)
 
    # TODO Trzeba zrobic wykrywanie czy w komorce jest liczba
    for i in range(len(cells) - 1, -1, -1):
        if not (number_inside(cells[i], 0.1)):
            cells.pop(i)     
 
    for cell in cells:
        cv2.imshow("Number", cell)
        getPrediction(cell, model)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
 
def main():
    model = load_model('Model1.h5')
    path = os.getcwd()
    os.chdir(path + "//sudoku_pics//")
    filenames = os.listdir()
    print(filenames)
    for filename in filenames:
        print(filename)
        img = cv2.imread(filename, 0)
        process(img, model)
 
 
if __name__ == '__main__':
    main()