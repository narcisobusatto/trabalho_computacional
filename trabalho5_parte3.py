import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats


FILE = 'Data_Iris.csv'

DIRECTORY = 'trabalho5_parte3'

FOLDER_CONFUSION_MATRIX = 'confusion_matrix'
FOLDER_CONFIDENCE_INTERVAL = 'confidence_interval'

CONFIDENCE1 = .95
CONFIDENCE2 = .997

NUMBER_EXECUTIONS = 10
TEST_SIZE = .2

fig, ax = plt.subplots()

def __init__():
    list_folders = [
        FOLDER_CONFUSION_MATRIX,
        FOLDER_CONFIDENCE_INTERVAL
    ]

    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    for folder in list_folders:
        if not os.path.exists(os.path.join(DIRECTORY, folder)):
            os.makedirs(os.path.join(DIRECTORY, folder))

def save_confidence_interval(acc_test, ci_length, classifier, confidence):
    """
    Method to save confidence interval to file

    :param acc_test: accuracy of test
    :param ci_lenght: confidence interval lenght
    :param classifier: classifier used
    :param confidence: confidence interval 

    """
    fig, ax = plt.subplots()
    ax.errorbar(acc_test, 0, xerr=ci_length, fmt="o")
    ax.set_xlim([0.85 if acc_test - ci_length > 0.85 else acc_test - ci_length - 0.05, 1.0])

    ax.set_yticks(np.arange(1))
    ax.set_xlabel("Prediction accuracy")

    plt.tight_layout()
    plt.grid(axis="x")
    plt.savefig(os.path.join(DIRECTORY, FOLDER_CONFIDENCE_INTERVAL, f"{classifier}_{str(confidence).replace('0.', '')}"))
    plt.close()

def save_confusion_matrix(data, classifier):
    """
    Method to save confusion matrix to file

    :param data: confusion matrix array
    :param classifier: classifier used
    
    """

    fig, ax = plt.subplots()
    cm_matrix = pd.DataFrame(data=data, index = ['SETOSA','VERSICOLOR','VIRGINICA'], 
                     columns = ['SETOSA','VERSICOLOR','VIRGINICA'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(os.path.join(DIRECTORY, FOLDER_CONFUSION_MATRIX, classifier))
    plt.close()



def gnb_classifier(data, test_size, column_label, print_classification_report=False, save_cm=False, save_ci=False):
    """
    Method to implement Gaussian Naïve-Bayes Classifier

    :param data: data (csv file)
    :param test_size: percentual of instances designated for test
    :column_label: column name used for label
    :param print_classification_report: boolean used to print classification report
    :param save_cm: boolean used to save confusion matrix
    :param save_ci: boolean used to save confidence interval

    :return float: accuracy or list: list of intervals (confidence)
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

    if print_classification_report:
        print(classification_report(y_test, y_pred))

    if save_cm:
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, 'gnb')

    if save_ci:
        interval = []
        for confidence in [CONFIDENCE1, CONFIDENCE2]:
            z_value = stats.norm.ppf((1 + confidence) / 2.0)

            acc_test = gnb.score(X_test, y_test)
            ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / y_test.shape[0])

            ci_lower = acc_test - ci_length if acc_test - ci_length > 0 else 0.0
            ci_upper = acc_test + ci_length if acc_test + ci_length < 1 else 1.0

            save_confidence_interval(acc_test, ci_length, 'gnb', confidence)

            interval.append([np.round(ci_lower, decimals=6), np.round(ci_upper, decimals=6)])

        return interval

    return np.round(accuracy_score(y_test, y_pred), decimals=6)


def lda_classifier(data, test_size, column_label, print_classification_report=False, save_cm=False, save_ci=False):
    """
    Method to implement Linear Discriminant Analysis

    :param data: data (csv file)
    :param test_size: percentual of instances designated for test
    :column_label: column name used for label
    :param print_classification_report: boolean used to print classification report
    :param save_cm: boolean used to save confusion matrix
    :param save_ci: boolean used to save confidence interval

    :return float: accuracy or list: list of intervals (confidence)
    """

    X = data.drop([column_label], axis=1)
    y = data[column_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(time.time()))

    clf = LinearDiscriminantAnalysis()

    clf.fit(
        X_train, 
        y_train
        )

    y_pred = clf.predict(X_test)

    if print_classification_report:
        print(classification_report(y_test, y_pred))

    if save_cm:
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, 'lda')

    if save_ci:
        interval = []
        for confidence in [CONFIDENCE1, CONFIDENCE2]:
            z_value = stats.norm.ppf((1 + confidence) / 2.0)

            acc_test = clf.score(X_test, y_test)
            ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / y_test.shape[0])

            ci_lower = acc_test - ci_length if acc_test - ci_length > 0 else 0.0
            ci_upper = acc_test + ci_length if acc_test + ci_length < 1 else 1.0

            save_confidence_interval(acc_test, ci_length, 'lda', confidence)

            interval.append([np.round(ci_lower, decimals=6), np.round(ci_upper, decimals=6)])

        return interval

    return np.round(accuracy_score(y_test, y_pred), decimals=6)

def qda_classifier(data, test_size, column_label, print_classification_report=False, save_cm=False, save_ci=False):
    """
    Method to implement Quadratic Discriminant Analysis

    :param data: data (csv file)
    :param test_size: percentual of instances designated for test
    :column_label: column name used for label
    :param print_classification_report: boolean used to print classification report
    :param save_cm: boolean used to save confusion matrix
    :param save_ci: boolean used to save confidence interval

    :return float: accuracy or list: list of intervals (confidence)
    """
    X = data.drop([column_label], axis=1)
    y = data[column_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(time.time()))

    clf = QuadraticDiscriminantAnalysis()

    clf.fit(
        X_train, 
        y_train
        )

    y_pred = clf.predict(X_test)

    if print_classification_report:
        print(classification_report(y_test, y_pred))

    if save_cm:
        cm = confusion_matrix(y_test, y_pred)
        save_confusion_matrix(cm, 'qda')

    if save_ci:
        interval = []
        for confidence in [CONFIDENCE1, CONFIDENCE2]:
            z_value = stats.norm.ppf((1 + confidence) / 2.0)

            acc_test = clf.score(X_test, y_test)
            ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / y_test.shape[0])

            ci_lower = acc_test - ci_length if acc_test - ci_length > 0 else 0.0
            ci_upper = acc_test + ci_length if acc_test + ci_length < 1 else 1.0

            save_confidence_interval(acc_test, ci_length, 'qda', confidence)

            interval.append([np.round(ci_lower, decimals=6), np.round(ci_upper, decimals=6)])

        return interval

    return np.round(accuracy_score(y_test, y_pred), decimals=6)


def main():
    data = pd.read_csv(FILE, header=None, comment='#')
    data = data[np.isfinite(data).all(1)]

    # 5.5
    print("============= LDA - CLASSIFICATION REPORT =============")
    accuracy_lda = lda_classifier(data, TEST_SIZE, 4, print_classification_report=True)
    print()
    print("============= QDA - CLASSIFICATION REPORT =============")
    accuracy_qda = qda_classifier(data, TEST_SIZE, 4, print_classification_report=True)
    print()
    print("********Comparative LDA x QDA********")
    print()
    print("Classifier\tAccuracy")
    print("-------------------------")
    print('LDA\t\t{:.6f}'.format(accuracy_lda))
    print('QDA\t\t{:.6f}'.format(accuracy_qda))
    print("-------------------------")
    print()

    # 5.6 and 5.7
    print("********Comparative GNB x LDA x QDA********")
    gnb = []
    lda = []
    qda = []

    for _ in range(0, NUMBER_EXECUTIONS):
        gnb.append(gnb_classifier(data, TEST_SIZE, 4, save_cm=True))
        lda.append(lda_classifier(data, TEST_SIZE, 4, save_cm=True))
        qda.append(qda_classifier(data, TEST_SIZE, 4, save_cm=True))

    print()
    print("Classifier\tMean\t\tVariance")
    print('---------------------------------------------')
    print('GNB\t\t{:.6f}\t{:.6f}'.format(np.round(np.mean(gnb), decimals=6), np.round(np.var(gnb), decimals=6)))
    print('LDA\t\t{:.6f}\t{:.6f}'.format(np.round(np.mean(lda), decimals=6), np.round(np.var(lda), decimals=6)))
    print('QDA\t\t{:.6f}\t{:.6f}'.format(np.round(np.mean(qda), decimals=6), np.round(np.var(qda), decimals=6)))
    print('---------------------------------------------')
    print()

    # 5.8
    gnb_interval = gnb_classifier(data, TEST_SIZE, 4, save_ci=True)
    lda_interval = lda_classifier(data, TEST_SIZE, 4, save_ci=True)
    qda_interval = qda_classifier(data, TEST_SIZE, 4, save_ci=True)

    print("********Confiance Interval********")
    print()
    print("Classifier\t\t95%\t\t\t\t99.7%")
    print('-----------------------------------------------------------------------')
    print('GNB\t\t{:.6f}-{:.6f}\t\t{:.6f}-{:.6f}'.format(gnb_interval[0][0], gnb_interval[0][1], gnb_interval[1][0], gnb_interval[1][1]))
    print('LDA\t\t{:.6f}-{:.6f}\t\t{:.6f}-{:.6f}'.format(lda_interval[0][0], lda_interval[0][1], lda_interval[1][0], lda_interval[1][1]))
    print('QDA\t\t{:.6f}-{:.6f}\t\t{:.6f}-{:.6f}'.format(qda_interval[0][0], qda_interval[0][1], qda_interval[1][0], qda_interval[1][1]))
    print('-----------------------------------------------------------------------')
    print()

if __name__ == '__main__':
    __init__()
    main()
