import cv2
import re
import os
from matplotlib import pyplot as plt
import numpy as np


DIRECTORY = 'trabalho1'
DIRECTORY_IMAGES = 'Images1'

FOLDER_COLORED = 'colored'
FOLDER_GRAYSCALE = 'grayscale'
FOLDER_REQUANTIZE_1 = 'requantize_1'
FOLDER_REQUANTIZE_2 = 'requantize_2'
FOLDER_REQUANTIZE_4 = 'requantize_4'

NROUNDS = 1

def __init__():

    list_folders = [
        FOLDER_COLORED,
        FOLDER_GRAYSCALE,
        FOLDER_REQUANTIZE_1,
        FOLDER_REQUANTIZE_2,
        FOLDER_REQUANTIZE_4,
    ]

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    for folder in list_folders:
        if not os.path.exists(os.path.join(DIRECTORY, folder)):
            os.makedirs(os.path.join(DIRECTORY, folder))

def save_image(img, folder, file):
    """
    Method to save image

    :param img: Image to save
    :param folder: folder where it will be saved
    :param file: image's filepath
    
    """
    filename = re.search('[ \w-]+?(?=\.)', file).group()
    cv2.imwrite(os.path.join(DIRECTORY, folder, f'{filename}.png'), img)

def get_image(file):
    """
    Method to view (save) image (in file)

    :param file: filepath to view

    """
    image = cv2.imread(file)
    save_image(image, FOLDER_COLORED, file)

def gray_scale(file):
    """
    Method to transform image in gray scale

    :param file: filepath to transform in gray scale

    :return matrix_NxM: image transformed to gray scale 
    """
    image = cv2.imread(file)
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    
    return imgGray

def get_grayscale_image(file):
    """
    Method to view imagem in gray scale

    :param file: filepath to view in gray scale

    """
    imgGray = gray_scale(file)
    save_image(imgGray, FOLDER_GRAYSCALE, file)

def requantize(nclusters, image):
    """
    Method to requantize image to N bits

    :param nclusters: amount of bits to requantize image
    :param image: matrix NxM containing image to requantize

    :return matrix NxM: image requantized
    """
    height, width = image.shape[:2]
    samples = np.zeros([height*width, 3], dtype = np.float32)
    count = 0
    
    for x in range(height):
        for y in range(width):
            samples[count] = image[x][y]
            count += 1
            
    compactness, labels, centers = cv2.kmeans(samples,
                                        nclusters, 
                                        None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
                                        NROUNDS, 
                                        cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    image2 = res.reshape((image.shape))

    return image2

def requantize_gray(nclusters, image):
    """
    Method to requantize image in gray scale to N bits

    :param nclusters: amount of bits to requantize image
    :param image: matrix NxM containing image in gray scale to requantize

    :return matrix NxM: image requantized
    """
    height, width = image.shape[:2]
    samples = np.zeros([height*width, 3], dtype = np.float32)
    count = 0
    
    for x in range(height):
        for y in range(width):
            samples[count] = 0.2989 * image[x][y][0] + \
                            0.5870 * image[x][y][1] + \
                            0.1140 * image[x][y][2]
            count += 1
            
    compactness, labels, centers = cv2.kmeans(samples,
                                        nclusters, 
                                        None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
                                        NROUNDS, 
                                        cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    image2 = res.reshape((image.shape))

    return image2

def requantize_image(nclusters, file):
    """
    Method to requantize image

    :param nclusters: amount of bits to requantize image
    :param file: filepath to requantize

    """
    requantize_folder = {
        1: FOLDER_REQUANTIZE_1,
        2: FOLDER_REQUANTIZE_2,
        4: FOLDER_REQUANTIZE_4
    }

    image = cv2.imread(file, 1)
    image2 = requantize(nclusters, image)    
    save_image(image2, requantize_folder[nclusters], file)
    
def get_files():
    """
    Method to get files in directory
    
    :return files: list of filepaths
    """
    files = []
    for dirname, _, filenames in os.walk(DIRECTORY_IMAGES):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    return files

def main():

    files = get_files()
    for file in files:

        # 1.1
        get_image(file)

        # 1.2 and 1.3
        get_grayscale_image(file)

        # 1.4
        requantize_image(4, file)

        # 1.5
        requantize_image(2, file)
        requantize_image(1, file)


if __name__ == '__main__':
    __init__()
    main()