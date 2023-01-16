import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


FILE_CSV = os.path.join('trabalho4', 'csv', 'trabalho4.csv')

def gnb_classifier(data, test_size, column_label):
    """
    Method to implement Gaussian Naïve-Bayes Classifier

    :param data: data (csv file created in Trabalho4)
    :param test_size: percentual of instances designated for test
    :column_label: column name used for label
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

    print("Number of mislabeled points out of a total {} points : {}, accuracy {:05.2f}%"
      .format(
          y_test.shape[0],
          (y_test != y_pred).sum(),
          100*(1-(y_test != y_pred).sum()/y_test.shape[0])
    ))
    print()

    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    print()

    print(classification_report(y_test, y_pred))
    print()

    y_pred_prob = gnb.predict_proba(X_test)
    print('Probabilities computed for each measure')
    print(y_pred_prob)
    print()

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(cm)

    
def main():
    # 5.1 and 5.2
    data = pd.read_csv(FILE_CSV)
    gnb_classifier(data, 0.2, 'label')


if __name__ == '__main__':
    main()
