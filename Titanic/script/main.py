# -*- encoding:utf-8 -*-

# https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV


def wrangle_data():
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    combine = [train_df, test_df]

    # 4.1  Correcting by dropping ticket,cabin
    print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    print "After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

    # 4.2  Creating title & drop name
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    # 4.3  Converting Sex
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # 4.4  Completing Age
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]

    # 4.5  Creating isAlone & drop Parch, SibSp
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    # 4.6  Creating Age*Class
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    # 4.7  Completing Embarked
    # fill null with the most common occurance
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    # 4.8  Converting categorical to numeric
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 4.9  Quick completing and converting a numeric feature
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    # combine = [train_df, test_df]
    return train_df,test_df


def KNN(X,y): # 15
    for i in range(1,30):
        knn = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # knn算法,k=5, 10倍交叉验证,cv(cross validation)=10
        # print scores  # 返回一个k维数组
        print i, scores.mean()   # 打印平均值

def SVM(X,y): #1 0.1
    C = [0.01,0.1,1,3,10]
    tol = [0.001,0.1,1,10]
    for c in C:
        for t in tol:
            svc = SVC(C=c, tol = t)
            scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')  # knn算法,k=5, 10倍交叉验证,cv(cross validation)=10
            # print scores  # 返回一个k维数组
            print c, t , scores.mean()  # 打印平均值


if __name__ == '__main__':
    train_df, test_df = wrangle_data()

    # Model
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    print X_train.shape, Y_train.shape, X_test.shape

    # KNN(X_train, Y_train)
    SVM(X_train,Y_train)
    # svm = SVC(C=1,tol=0.1)
    # svm.fit(X_train,Y_train)
    # y_pred = svm.predict(X_test)
    # print y_pred
    #
    # file = open('res.csv','wb')
    # for i in y_pred:
    #     file.write(str(i))
    #     file.write('\n')
    # file.close()

    # plt.show()

