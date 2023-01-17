import scipy
from scipy import stats
import skimage
import numpy as np
import pandas as pd
import os
import cv2


EPSILON = 2.2e-16
BINS = 16
DECIMALS = 4

DIRECTORY = 'trabalho4'
FILE_CSV = 'trabalho4.csv'
DIRECTORY_IMAGE = 'Image dataset'
FOLDER_CSV = 'csv'

def __init__():
    list_folders = [
        FOLDER_CSV
    ]

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    for folder in list_folders:
        if not os.path.exists(os.path.join(DIRECTORY, folder)):
            os.makedirs(os.path.join(DIRECTORY, folder))

def minimun(arr):
    """
    Method to calcule minimun from array

    :param arr: array

    :return float: minimun
    """

    return np.round(np.min(arr), decimals=DECIMALS)

def maximun(arr):
    """
    Method to calcule maximun from array

    :param arr: array

    :return float: maximun
    """

    return np.round(np.max(arr), decimals=DECIMALS)

def mean(arr):
    """
    Method to calcule mean from array

    :param arr: array

    :return float: mean
    """
    return np.round(np.mean(arr), decimals=DECIMALS)

def std(arr):
    """
    Method to calcule standard deviation from array

    :param arr: array

    :return float: standard deviation
    """

    return np.round(np.std(arr), decimals=DECIMALS)

def square_dynamic_range(arr):
    """
    Method to calcule square dynamic range from array

    :param arr: array

    :return float: square dynamic range
    """

    return np.round((maximun(arr) - minimun(arr)) ** 2, decimals=DECIMALS)

def var(arr):
    """
    Method to calcule variance deviation from array

    :param arr: array

    :return float: variance
    """

    return np.round(np.var(arr, axis=0), decimals=DECIMALS)

def median(arr):
    """
    Method to calcule median deviation from array

    :param arr: array

    :return float: median
    """

    return np.round(np.median(arr), decimals=DECIMALS)

def skewness(arr):
    """
    Method to calcule skewness from array

    :param arr: array

    :return float: skewness
    """
    
    return np.round(stats.skew(arr, axis = None), decimals=DECIMALS)

def kurtosis(arr):
    """
    Method to calcule kurtosis from array

    :param arr: array

    :return float: kurtosis
    """

    return np.round(stats.kurtosis(arr, axis = None), decimals=DECIMALS)
        
def percentile(arr, percentile):
    """
    Method to calcule percentile from array

    :param arr: array
    :param percentile: percentile of interesses

    :return float: percentile
    """

    return np.round(np.percentile(arr,percentile), decimals=DECIMALS)

def interquartile_range(arr):
    """
    Method to calcule interquartile range from array

    :param arr: array

    :return float: interquartile range
    """

    return np.round(np.subtract(*np.percentile(arr, [75, 25])), decimals=DECIMALS)

def shannon_entropy(arr):
    """
    Method to calcule Shannon entropy from array

    :param arr: array

    :return float: Shannon entropy
    """

    return np.round(skimage.measure.shannon_entropy(arr), decimals=DECIMALS)

def bins_entropy(image):
    """
    Method to calcule Bins entropy from image

    :param image: Image

    :return float: Bins entropy
    """

    N, M = image.shape                          
    histRange = [0,256]
    hist = cv2.calcHist([np.float32(image)], [0], None, [BINS], histRange) 
    histogram = hist.flatten()
    hist_sum = sum(histogram)
    pmf = histogram / hist_sum 
    bins_entropy = 0
    for i in range(BINS):
        bins_entropy -= pmf[i] * np.log2(pmf[i] + EPSILON)
    return np.round(bins_entropy, decimals=DECIMALS)

def norm_energy(image):
    """
    Method to calcule normalized energy from image

    :param image: Image

    :return float: normalized energy
    """

    N, M = image.shape
    return np.round(np.sum( image**2, axis = None ) / float(N * M), decimals=DECIMALS)

def root_means_square(image):
    """
    Method to calcule root means square (RMS) from image

    :param image: Image

    :return float: RMS
    """

    return np.round(np.sqrt(norm_energy(image)), decimals=DECIMALS)

def mean_absolute_deviation(image):
    """
    Method to calcule mean absolute deviation (MAD) from image

    :param image: Image

    :return float: MAD
    """

    N, M = image.shape
    return np.round(np.sum( np.abs(image - mean(image.reshape(-1))), axis = None ) / float(N * M), decimals=DECIMALS)

def robust_mean_absolute_deviation(image):
    """
    Method to calcule robust mean absolute deviation (rMAD) from image

    :param image: Image

    :return float: rMAD
    """
    N,M = image.shape
    image_p900 = np.percentile(image, 90.0)   
    image_p100 = np.percentile(image, 10.0)
    Np_10_90 = 0 
    rMAD = 0 

    mean_10_90 = mean([i for i in image.reshape(-1) if i >= image_p100 and i <= image_p900])
    
    for n in range(N):
        for m in range(M):
            if( (image[n,m]) >= image_p100) and (image[n,m] <= image_p900):
                Np_10_90 += 1
                rMAD += np.abs(image[n,m] - mean_10_90)               
    rMAD /= Np_10_90
    return np.round(rMAD, decimals=DECIMALS)

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


def get_label(file):
    """
    Method to identify image label

    :param file: File

    :return int: label [0-Alzheimer, 1-COVID, 2-Brazilian_Seeds, 3-Brazilian_Leaves, 4-Skin_Cancer]
    """

    labels = {
        'Alzheimer': 0,
        'COVID': 1,
        'Brazilian_Seeds': 2,
        'Brazilian_Leaves': 3,
        'Skin_Cancer': 4
    }

    for k in labels.keys():
        if k in file:
            return labels[k]

def get_features(f):
    """
    Method to load features from file

    :param file: File

    :return dict: features
    """
    image = gray_scale(f)
    img = image.reshape(-1)
        
    features = {
        'minimun': minimun(img),
        'maximun': maximun(img),
        'sdr': square_dynamic_range(img),
        'mean': mean(img),
        'std': std(img),
        'variance': var(img),
        'skewness': skewness(img),
        'kurtosis': kurtosis(img),
        '7_5percentile': percentile(img, 7.5),
        '15percentile': percentile(img, 15),
        '50percentile': percentile(img, 50),
        '85percentile': percentile(img, 85),
        '92_5percentile': percentile(img, 92.5),
        'median': median(img),
        'interquatile_range': interquartile_range(img),
        'shannon_entropy': shannon_entropy(img),
        'bins_entropy': bins_entropy(image),
        'norm_energy': norm_energy(image),
        'rmsv': root_means_square(image),
        'mad': mean_absolute_deviation(image),
        'rmad': robust_mean_absolute_deviation(image),
        'label': get_label(f)
    }

    return features

def get_files():
    """
    Method to get files in directory
    
    :return files: list of filepaths
    """
    files = []
    for dirname, _, filenames in os.walk(DIRECTORY_IMAGE):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    return files


def main():
    columns = ['minimun', 'maximun', 'sdr', 'mean', 'std', 'variance', 'skewness', 'kurtosis', '7_5percentile', '15percentile', \
               '50percentile', '85percentile', '92_5percentile', 'median', 'interquatile_range', 'shannon_entropy', 'bins_entropy', \
               'norm_energy', 'rmsv', 'mad', 'rmad', 'label']

    df = pd.DataFrame(columns=columns)

    files = get_files()
    for file in files:
        features = get_features(file)
        df_temp = pd.DataFrame([features])
        df = df.append(df_temp)
    df.to_csv(os.path.join(DIRECTORY, FOLDER_CSV, FILE_CSV), index=False)


if __name__ == '__main__':
    __init__()
    main()