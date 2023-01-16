import cv2
import re
import os
from matplotlib import pyplot as plt
import numpy as np
import builtins


DIRECTORY = 'trabalho2'
DIRECTORY_IMAGES = 'Images1'

FOLDER_PMF = 'pmf'

NROUNDS = 1
DECIMALS = 4

def __init__():
    list_folders = [
        FOLDER_PMF
    ]

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    for folder in list_folders:
        if not os.path.exists(os.path.join(DIRECTORY, folder)):
            os.makedirs(os.path.join(DIRECTORY, folder))

def get_image(file):
    """
    Method to get an image by filepath

    :return matrix: image
    """
    return cv2.imread(file)


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

def get_pmf(image):
    """
    Method to calcule pmf from image

    :param image: Image to calcule

    :return val, pmf: occurrences (val) and their probabilities (pmf)
    
    """
    img = image.reshape(-1)
    val, cnt = np.unique(img, return_counts=True)
    pmf = cnt / img.shape[0] 

    return val, pmf

#def moment(val, pmf, moment):
#    return np.sum([v ** moment for v in val] * pmf

#def mean(val, pmf):
#    return moment(val, pmf, 1)

#def variance(val, pmf):
#    return moment(val, pmf, 2) - moment(val, pmf, 1) ** 2

#def skewness(val, pmf):
#    mu = moment(val, pmf, 1)
#    sigma = np.sqrt(variance(val, pmf))
#    return (moment(val, pmf, 3) - 3 * mu * (sigma ** 2) - mu ** 3 ) / (sigma ** 3)

#def kurtosis(val, pmf):
#    mu = moment(val, pmf, 1)
#    sigma = np.sqrt(variance(val, pmf))
#    return np.sum([(v - mu) ** 4 for v in val] * pmf) / (sigma ** 4)

def mean(arr):
    """
    Method to calcule mean from array

    :param arr: array

    :return float: mean
    """

    return np.round(builtins.sum(arr) / len(arr), decimals=DECIMALS)

def variance(arr):
    """
    Method to calcule variance from array

    :param arr: array

    :return float: variance
    """
    mu = mean(arr)
    desv = ((a - mu) ** 2 for a in arr)
    return np.round(builtins.sum(desv) / len(arr), decimals=DECIMALS)

def skewness(arr):
    """
    Method to calcule skewness from array

    :param arr: array

    :return float: skewness
    """
    mu = mean(arr)
    sigma = np.sqrt(variance(arr))
    skew_part = (np.divide(a-mu, sigma) ** 3 for a in arr)
    return np.round(builtins.sum( skew_part )/len(arr), decimals=DECIMALS)

def kurtosis(arr):
    """
    Method to calcule kurtosis from array

    :param arr: array

    :return float: kurtosis
    """
    mu = mean(arr)
    sigma = np.sqrt(mean(arr))
    return np.round(builtins.sum( np.divide(a-mu, sigma) ** 4 for a in arr)/len(arr), decimals=DECIMALS)

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

def save_pmf(x, y, folder, file):
    """
    Method to save pmf to file

    :param x: x-axis values
    :param y: y-axis values
    :param folder: destination folder
    :param file: reference file
    """

    filename = re.search('[ \w-]+?(?=\.)', file).group()

    fig, ax = plt.subplots()
    plt.bar(x, y)
    plt.savefig(os.path.join(DIRECTORY, folder, filename), dpi=400)


def main():

    files = get_files()
    for file in files:

        filename = re.search('([^\/]*)\.[^.]*$', file).group()
        print(f"File: {filename}")

        # 2.1
        imgGray = requantize_gray(4, get_image(file))
        val, pmf = get_pmf(imgGray)
        save_pmf(val, pmf, FOLDER_PMF, file)

        img = imgGray.reshape(-1)

        # 2.2
        print(f'Mean: {mean(img)}')
        print(f'Variance: {variance(img)}')

        # 2.3
        print(f'Skewness: {skewness(img)}')
        print(f'Kurtosis: {kurtosis(img)}')
        print("====================================")
        print()


if __name__ == '__main__':
    __init__()
    main()
