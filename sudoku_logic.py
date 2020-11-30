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
import copy

class AlgoManager():
    def __init__(self, filename):
        self.im = cv2.imread(filename)
        #cv2.imshow("img", self.im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        self.model = load_model('Model1.h5')
        self.detector = GridDetector()
        self.pics = []
        # Dodaje do listy pics obrazek wejsciowy
        self.pics.append(self.im)
        self.result = None
        self.res_grid_final, grid_corners, _ = self.detector.extract_grids(self.im)
        # Dodaje do listy obrazek z narysowanym konturem
        self.pics.append(self.printContourOn(self.im, grid_corners))
        
        if self.res_grid_final is not None:
            # Dodaje do listy pics obrazek po dokonaniu projekcji
            self.pics.append(self.res_grid_final) 
            #cv2.imshow("perspective", self.res_grid_final)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            numbers = extract_cells(self.res_grid_final, self.model)
            self.result = self.compareDigits(numbers, filename)
            # Dodaje do listy pics obrazek finalny
            self.pics.append(self.printNumbersOn(copy.copy(self.res_grid_final), numbers, grid_corners))
            
    # Metoda naklada na wejsciowy obrazek rozpoznane cyfry
    def printNumbersOn(self, im, numbers, grid_corners):
    # GRID CORNERS: NW, NE, SE, SW 
        # Odwrotne rzutowanie
        final_pts = np.array(
            [[0, 0], [target_w_grid - 1, 0],
             [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
            dtype=np.float32)
        transfo_mat = cv2.getPerspectiveTransform(final_pts, grid_corners.astype(np.float32))
        for i in range(9):
            for j in range(9):
                if numbers[i][j] != -1:
                    position = (target_w_grid//9 * j + 30, target_h_grid//9 * i + 30)
                    #position = np.dot(np.array(transfo_mat), np.array([target_w_grid//9 * j, target_h_grid//9 * i, 1]))
                    #position = (int(position[0]), int(position[1]))
                    cv2.putText(im, str(numbers[i][j]), position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0, 255), 2) 
        #im = cv2.resize(im, (600, 600), interpolation = cv2.INTER_AREA)
        #cv2.imshow("final", im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return im
        
        
    # Metoda naklada na wejsciowy obrazek kontur
    def printContourOn(self, im, grid_corners):
        ax = plt.gca()
        ax.imshow(im)
        origin = np.array((0, im.shape[1]))
        ax.plot(grid_corners[0][0], grid_corners[0][1], 'bo')
        ax.plot(grid_corners[1][0], grid_corners[1][1], 'bo')
        ax.plot(grid_corners[2][0], grid_corners[2][1], 'bo')
        ax.plot(grid_corners[3][0], grid_corners[3][1], 'bo')
        ax.set_xlim(origin)
        ax.set_ylim((im.shape[0], 0))
        ax.set_axis_off()
        plt.tight_layout()
    
        plt.savefig("tmp.png")
        img_contours = cv2.imread("tmp.png")
        plt.close()
        #cv2.imshow("contours", img_contours)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return img_contours
    
    # Metoda oblicza result (procent rozpoznanych cyfr)
    def compareDigits(self, numbers, filename):
        path = os.getcwd()
        digits_recognized = 0
        total_digits = 0
        templateFile = open(path + "\\" + filename[:-4] + ".txt")
        for i, line in enumerate(templateFile):
            digits = line.replace(" ", "")
            for j, digit in enumerate(digits[:-1]):
                if numbers[i][j] != -1:
                    total_digits += 1
                    print(digit, numbers[i][j])
                    if int(digit) == numbers[i][j]:
                        digits_recognized += 1
        templateFile.close()
        print("recognized:", digits_recognized)
        print("total digits:", total_digits)
        return digits_recognized / total_digits
        
    def getPictures(self):
        return self.pics
    
    def getResult(self):
        return self.result


# ----PREPRO BIG IMAGE----#
block_size_big = 41
block_size_webcam_big = 21
mean_sub_big = 15
mean_sub_webcam_big = 5

# display_prepro_big = True
display_prepro_big = False

# ----GRID COUNTOURS----#
ratio_lim = 2
smallest_area_allow = 75000
approx_poly_coef = 0.1

# ----GRID UPDATE AND SIMILARITY----#
lim_apparition_not_solved = 12
lim_apparition_solved = 60
same_grid_dist_ratio = 0.05
target_h_grid, target_w_grid = 450, 450

class GridDetector:
    def __init__(self, display=False):
        self.__display = display

    def extract_grids(self, frame):
        # Get a threshed image which emphasize lines
        threshed_img = self.thresh_img(frame)

        # Look for grids corners
        grid_corners = self.look_for_grids_corners(threshed_img)
        if grid_corners is None:
            return None
            
        # Use grids corners to unwrap img !
        unwraped_grid_list, transfo_matrix = self.unwrap_grids(frame, grid_corners)
        return unwraped_grid_list, grid_corners, transfo_matrix

    @staticmethod
    def thresh_img(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))

        blurred = cv2.GaussianBlur(gray_enhance, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       block_size_big, mean_sub_big)

        thresh_not = cv2.bitwise_not(thresh)

        kernel_close = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)  # Delete space between line
        dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)  # Delete space between line
        return dilate

    @staticmethod
    def unwrap_grids(frame, points_grid):
        final_pts = np.array(
            [[0, 0], [target_w_grid - 1, 0],
             [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
            dtype=np.float32)
        print(points_grid)
        transfo_mat = cv2.getPerspectiveTransform(points_grid.astype(np.float32), final_pts)
        undistorted_grid = cv2.warpPerspective(frame, transfo_mat, (target_w_grid, target_h_grid))
        transfo_matrix = np.linalg.inv(transfo_mat)
        return undistorted_grid, transfo_matrix

    @staticmethod
    def look_for_grids_corners(img_lines):
        contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        biggest_area = cv2.contourArea(contours[0])

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < smallest_area_allow:
                break
            if area > biggest_area / ratio_lim:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, approx_poly_coef * peri, True)
                if len(approx) == 4:
                    best_contour = approx
        return np.array([best_contour[0][0], best_contour[3][0], best_contour[2][0], best_contour[1][0]])
 
def getPrediction(image, model):

    ## Preparing img so it can be used by model
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    #img = cv2.bitwise_not(image.copy())
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
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    img2 = img2.reshape(1, 28, 28, 1)
    ## Using model
    predictions = model.predict(img2)
    classIndex = model.predict_classes(img2)
    probabilityValue = np.amax(predictions)

    return predictions, classIndex, probabilityValue

def extract_cells(projected_img, model):
    # Cropping out cells from grid
    x_diff = projected_img.shape[1] // 9
    y_diff = projected_img.shape[0] // 9
    cellso = []
    cells = []
    for i in range(9):
        for j in range(9):
            lower_bound_y = i * y_diff
            upper_bound_y = (i + 1) * y_diff
            lower_bound_x = j * x_diff
            upper_bound_x = (j + 1) * x_diff

            cell = projected_img[lower_bound_y:upper_bound_y, lower_bound_x:upper_bound_x]
            #cv2.imshow("first", cell)
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 5, 2)
            gray = cv2.bitwise_not(gray)
            #cv2.imshow("second", gray)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cellso.append(cell)
            cells.append(gray)

    # Sprawdzamy czy w danej wycietej komorce jest liczba zliczajac liczbe bialych pixeli w srodku komorki
    for i in range(len(cells) - 1, -1, -1):
        if not (number_inside(cells[i], 0.1)):
            #cv2.imshow("second", cells[i])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cells[i] = []
            cellso[i] = []

    numbers = []
    for i in range(9):
        numbers.append([])
        for j in range(9):
            cell = cellso[i * 9 + j]
            if cell == []:
                numbers[i].append(-1)
            else:
                numbers[i].append(getPrediction(cell, model)[1][0])
    for n in numbers:
        print(n)
    return numbers

# Returns True if there is number in cell or False if there is not (needs to be upgraded)
def number_inside(cell, percent):
    h, w = cell.shape
    searching_x = w // 2
    searching_y = h // 2
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
 

 