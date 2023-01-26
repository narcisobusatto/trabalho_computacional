import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


DIRECTORY = 'trabalho5_parte2'

FOLDER_MEASURE = 'measures'

NUMBER_EXECUTIONS = 10

FILE_CSV = os.path.join('trabalho4', 'csv', 'trabalho4.csv')

def __init__():
    list_folders = [
        FOLDER_MEASURE
    ]

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    for folder in list_folders:
        if not os.path.exists(os.path.join(DIRECTORY, folder)):
            os.makedirs(os.path.join(DIRECTORY, folder))

def gnb_classifier(data, test_size, column_label):
    """
    Method to implement Gaussian Naïve-Bayes Classifier

    :param data: data (csv file created in Trabalho4)
    :param test_size: percentual of instances designated for test
    :param column_label: column name used for label

    return float: accuracy score
    """

    X = data.drop([column_label], axis=1)
    y = data[column_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(time.time()))

    gnb = GaussianNB()

    gnb.fit(
        X_train, 
        y_train
        )

    y_pred = gnb.predict(X_test)

    return accuracy_score(y_test, y_pred)

def save_plot(dataframe, column_reference, measure, kind):
    """
    Method to save a plot

    :param dataframe: data (pandas)
    :param column_reference: column defined as x-axis
    :param measure: measure defined as y-axis
    :param kind: plot's kind
    """

    dataframe.plot(x=column_reference, y=measure, kind=kind)
    plt.savefig(os.path.join(DIRECTORY, FOLDER_MEASURE, f'{measure}.png'))


def main():
    # 5.3 and 5.4
    data = pd.read_csv(FILE_CSV)
    df = pd.DataFrame(columns=['test_size', 'mean', 'var'])

    for test_size in [x * 0.01 for x in range(20, 100, 5)]:
        classifier_measures = []
        for _ in range(0,NUMBER_EXECUTIONS):
            classifier_measures.append(gnb_classifier(data, test_size, 'label'))

        d = {
            'test_size': test_size,
            'mean': np.mean(classifier_measures),
            'var': np.var(classifier_measures)
        }

        df = df.append(pd.DataFrame([d]))
            
    save_plot(df, 'test_size', 'mean', 'line')
    save_plot(df, 'test_size', 'var', 'line')
    

if __name__ == '__main__':
    __init__()
    main()
