# -*- encoding:utf-8 -*-

# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

import pandas as pd
import numpy as np
import sklearn

import random
import time

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# from pandas.tools.plotting import scatter_matrix


def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 6}
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)


def clean_data(train,test):
    combine = [train,test]

    for data in combine:
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

        drop_column = ['PassengerId', 'Cabin', 'Ticket']
        data.drop(drop_column, axis=1, inplace=True)

        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

        data['IsAlone'] = 1  # initialize to yes/1 is alone
        data['IsAlone'].loc[data['FamilySize'] > 1] = 0
        data['Title'] = data['Name'].str.split(", ", expand=True)[1]\
                                    .str.split(".", expand=True)[0]
        # cut by frequency
        data['FareBin'] = pd.qcut(data['Fare'], 4)
        # cut by bin-width
        data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

    small = 10
    title_names = (train['Title'].value_counts() < small)
    # print train['Title'].value_counts()
    # print title_names
    train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)


def convert_formats(combine):
    label = LabelEncoder()
    for data in combine:
        data['Sex_Code'] = label.fit_transform(data['Sex'])
        data['Embarked_Code'] = label.fit_transform(data['Embarked'])
        data['Title_Code'] = label.fit_transform(data['Title'])
        data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])
        data['FareBin_Code'] = label.fit_transform(data['FareBin'])


def explore_data(train, feat_sel_col, tar_col):
    # ############# 7.1 Discrete Variable Correlationusing group by
    for feat in feat_sel_col:
        if train[feat].dtype != 'float64':
            print 'Survival Correlation by:', feat
            print train[[feat, tar_col[0]]].groupby(feat, as_index=False).mean()
            print '-' * 10, '\n'
    print pd.crosstab(train['Title'], train[tar_col[0]])

    # ############# 7.2 Graph distribution of quantitative data
    plt.figure(figsize=[16, 12])
    # ### 7.21 boxplot
    plt.subplot(231)
    plt.boxplot(x=train['Fare'], showmeans=True, meanline=True)
    plt.title('Fare Boxplot')
    plt.ylabel('Fare ($)')

    plt.subplot(232)
    plt.boxplot(train['Age'], showmeans=True, meanline=True)
    plt.title('Age Boxplot')
    plt.ylabel('Age (Years)')

    plt.subplot(233)
    plt.boxplot(train['FamilySize'], showmeans=True, meanline=True)
    plt.title('Family Size Boxplot')
    plt.ylabel('Family Size (#)')

    # ### 7.22 hist
    plt.subplot(234)
    plt.hist(x=[train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']],
             stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Fare Histogram by Survival')
    plt.xlabel('Fare ($)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(235)
    plt.hist(x=[train[train['Survived'] == 1]['Age'], train[train['Survived'] == 0]['Age']],
             histtype='bar', color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Age Histogram by Survival')
    plt.xlabel('Age (Years)')
    plt.ylabel('# of Passengers')
    plt.legend()

    plt.subplot(236)
    plt.hist(x=[train[train['Survived'] == 1]['FamilySize'], train[train['Survived'] == 0]['FamilySize']],
             stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
    plt.title('Family Size Histogram by Survival')
    plt.xlabel('Family Size (#)')
    plt.ylabel('# of Passengers')
    plt.legend()

    # ############# 7.3 Use seaborn graphics for multi-variable comparison
    fig, saxis = plt.subplots(2, 3, figsize=(16, 12))

    sns.barplot(x='Embarked', y='Survived', data=train, ax=saxis[0, 0])
    sns.barplot(x='Pclass', y='Survived', order=[1, 2, 3], data=train, ax=saxis[0, 1])
    sns.barplot(x='IsAlone', y='Survived', order=[1, 0], data=train, ax=saxis[0, 2])

    sns.pointplot(x='FareBin', y='Survived', data=train, ax=saxis[1, 0])
    sns.pointplot(x='AgeBin', y='Survived', data=train, ax=saxis[1, 1])
    sns.pointplot(x='FamilySize', y='Survived', data=train, ax=saxis[1, 2])

    # ############### 7.4 Graph distribution of qualitative data
    # ### 7.41 Pclass
    fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(14, 12))

    sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=train, ax=axis1)
    axis1.set_title('Pclass vs Fare Survival Comparison')

    sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train, split=True, ax=axis2)
    axis2.set_title('Pclass vs Age Survival Comparison')

    sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=train, ax=axis3)
    axis3.set_title('Pclass vs Family Size Survival Comparison')

    # ### 7.42 Sex
    fig, qaxis = plt.subplots(1, 3, figsize=(14, 12))

    sns.barplot(x='Sex', y='Survived', hue='Embarked', data=train, ax=qaxis[0])
    axis1.set_title('Sex vs Embarked Survival Comparison')

    sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train, ax=qaxis[1])
    axis1.set_title('Sex vs Pclass Survival Comparison')

    sns.barplot(x='Sex', y='Survived', hue='IsAlone', data=train, ax=qaxis[2])
    axis1.set_title('Sex vs IsAlone Survival Comparison')

    # ############### 7.5 More side-by-side comparisons
    fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=(14, 12))

    # ### 7.51 how does family size factor with sex & survival compare
    sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=train,
                  palette={"male": "blue", "female": "pink"},
                  markers=["*", "o"], linestyles=["-", "--"], ax=maxis1)

    # ### 7.52 how does class factor with sex & survival compare
    sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,
                  palette={"male": "blue", "female": "pink"},
                  markers=["*", "o"], linestyles=["-", "--"], ax=maxis2)

    # ### 7.53 how does embark port factor with class, sex, and survival compare
    e = sns.FacetGrid(train, col='Embarked')
    e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep')
    e.add_legend()

    # ### 7.54 plot distributions of age of passengers who survived or did not survive
    a = sns.FacetGrid(train, hue='Survived', aspect=4)
    a.map(sns.kdeplot, 'Age', shade=True)
    a.set(xlim=(0, train['Age'].max()))
    a.add_legend()

    # ### 7.55 histogram comparison of sex, class, and age by survival
    h = sns.FacetGrid(train, row='Sex', col='Pclass', hue='Survived')
    h.map(plt.hist, 'Age', alpha=.75)
    h.add_legend()

    # ### 7.56 pair plots of entire dataset
    pp = sns.pairplot(train, hue='Survived', palette='deep', size=1.2, diag_kind='kde',
                      diag_kws=dict(shade=True),plot_kws=dict(s=10))
    pp.set(xticklabels=[])

    # ############### 7.6 Correlation heatmap of dataset
    correlation_heatmap(train)


def try_model(train, tar_col, feat_bin_col):
    # Machine Learning Algorithm (MLA) Selection and Initialization
    MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(),

        # GLM
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        XGBClassifier()
    ]

    # split dataset in cross-validation with this splitter class
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)

    # create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean',
                   'MLA Test Accuracy Mean','MLA Test Accuracy 3*STD', 'MLA Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # here, we don't do it correctly, we just have a try
    # create table to compare MLA predictions
    MLA_predict = train[tar_col]
    row_index = 0
    for alg in MLA:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        # score model with cross validation
        cv_results = model_selection.cross_validate(alg, train[feat_bin_col],
                                                    train[tar_col], cv=cv_split)
        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean,
        # should statistically capture 99.7% of the subsets
        # let's know the worst that can happen!
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3

        # save MLA predictions - see section 6 for usage
        alg.fit(train[feat_bin_col], train[tar_col])
        MLA_predict[MLA_name] = alg.predict(train[feat_bin_col])

        row_index += 1

    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
    # print MLA_compare

    sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='m')
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.ylabel('Algorithm')

    print MLA_predict.corr()
    correlation_heatmap(MLA_predict)


def tune_dt(train,feat_bin_col,tar_col):
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)
    # ############# 9.1 Tune hyper parameters
    # ### 9.11 base model
    dtree = tree.DecisionTreeClassifier(random_state=0)
    base_results = model_selection.cross_validate(dtree, train[feat_bin_col], train[tar_col], cv=cv_split)
    dtree.fit(train[feat_bin_col], train[tar_col])

    print('BEFORE DT Parameters: ', dtree.get_params())
    print("BEFORE DT Training w/bin score mean: {:.2f}".format(base_results['train_score'].mean() * 100))
    print("BEFORE DT Test w/bin score mean: {:.2f}".format(base_results['test_score'].mean() * 100))
    print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}".format(base_results['test_score'].std() * 100 * 3))
    # print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))
    print('-' * 10)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        # 'splitter': ['best', 'random'],
        'max_depth': [2, 4, 6, 8, 10, None],
        #  'min_samples_split': [2,5,10,.03,.05],
        #  'min_samples_leaf': [1,5,10,.03,.05],
        #  'max_features': [None, 'auto'],
        'random_state': [0]
    }

    # print(list(model_selection.ParameterGrid(param_grid)))

    # ### 9.12
    tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
                                              param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    tune_model.fit(train[feat_bin_col], train[tar_col])

    # print(tune_model.cv_results_.keys())
    # print(tune_model.cv_results_['params'])
    print('AFTER DT Parameters: ', tune_model.best_params_)
    print(tune_model.cv_results_['mean_train_score'])
    print("AFTER DT Training w/bin score mean: {:.2f}".format(
        tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
    # print(tune_model.cv_results_['mean_test_score'])
    print("AFTER DT Test w/bin score mean: {:.2f}".format(
        tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
    print("AFTER DT Test w/bin score 3*std: +/- {:.2f}".format(
        tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
    print('-' * 10)

    # # duplicates gridsearchcv
    # tune_results = model_selection.cross_validate(tune_model, train[feat_bin_col],
    #                                               train[tar_col], cv=cv_split)
    # print tune_results
    # print('AFTER DT Parameters: ', tune_model.best_params_)
    # print("AFTER DT Training w/bin set score mean: {:.2f}". format(
    #     tune_results['train_score'].mean()*100))
    # print("AFTER DT Test w/bin set score mean: {:.2f}".format(
    #     tune_results['test_score'].mean()*100))
    # print("AFTER DT Test w/bin set score min: {:.2f}".format(
    #     tune_results['test_score'].min()*100))
    # print('-'*10)

    # ############# 9.2 Tune feature selection
    # ### 9.21 base model
    print('BEFORE DT RFE Training Shape Old: ', train[feat_bin_col].shape)
    print('BEFORE DT RFE Training Columns Old: ', train[feat_bin_col].columns.values)

    print("BEFORE DT RFE Training w/bin score mean: {:.2f}".format(
        base_results['train_score'].mean() * 100))
    print("BEFORE DT RFE Test w/bin score mean: {:.2f}".format(
        base_results['test_score'].mean() * 100))
    print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}".format(
        base_results['test_score'].std() * 100 * 3))
    print('-' * 10)

    # ### 9.22 feature selection
    dtree_rfe = feature_selection.RFECV(dtree, step=1, scoring='accuracy', cv=cv_split)
    dtree_rfe.fit(train[feat_bin_col], train[tar_col])

    X_rfe = train[feat_bin_col].columns.values[dtree_rfe.get_support()]
    rfe_results = model_selection.cross_validate(dtree, train[X_rfe], train[tar_col], cv=cv_split)

    # print(dtree_rfe.grid_scores_)
    print('AFTER DT RFE Training Shape New: ', train[X_rfe].shape)
    print('AFTER DT RFE Training Columns New: ', X_rfe)

    print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(
        rfe_results['train_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(
        rfe_results['test_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(
        rfe_results['test_score'].std() * 100 * 3))
    print('-' * 10)

    # ### 9.23 tune rfe model
    rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
                                                  param_grid=param_grid, scoring='roc_auc', cv=cv_split)
    rfe_tune_model.fit(train[X_rfe], train[tar_col])

    # print(rfe_tune_model.cv_results_.keys())
    # print(rfe_tune_model.cv_results_['params'])
    print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
    # print(rfe_tune_model.cv_results_['mean_train_score'])
    print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}".format(
        rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_] * 100))
    # print(rfe_tune_model.cv_results_['mean_test_score'])
    print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}".format(
        rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_] * 100))
    print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}".format(
        rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_] * 100 * 3))
    print('-' * 10)

    # ### visualize tree
    # import graphviz
    # dot_data = tree.export_graphviz(dtree, out_file=None,
    #                                 feature_names=feat_bin_col, class_names=True,
    #                                 filled=True, rounded=True)
    # graph = graphviz.Source(dot_data)
    # graph.render('test.gv', view=True)


def validate_implement(train,feat_bin_col,tar_col,test):
    # ############### 1
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)
    vote_est = [
        # Ensemble Methods
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc', ensemble.ExtraTreesClassifier()),
        ('gbc', ensemble.GradientBoostingClassifier()),
        ('rfc', ensemble.RandomForestClassifier()),

        # Gaussian Processes
        ('gpc', gaussian_process.GaussianProcessClassifier()),

        # GLM
        ('lr', linear_model.LogisticRegressionCV()),

        # Navies Bayes
        ('bnb', naive_bayes.BernoulliNB()),
        ('gnb', naive_bayes.GaussianNB()),

        # Nearest Neighbor
        ('knn', neighbors.KNeighborsClassifier()),

        # SVM
        ('svc', svm.SVC(probability=True)),

        # xgboost
        ('xgb', XGBClassifier())
    ]

    # Hard Vote or majority rules
    vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    vote_hard_cv = model_selection.cross_validate(vote_hard, train[feat_bin_col], train[tar_col], cv=cv_split)
    vote_hard.fit(train[feat_bin_col], train[tar_col])

    print("Hard Voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean() * 100))
    print("Hard Voting Test w/bin score mean: {:.2f}".format(vote_hard_cv['test_score'].mean() * 100))
    print("Hard Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_hard_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # Soft Vote or weighted probabilities
    vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    vote_soft_cv = model_selection.cross_validate(vote_soft, train[feat_bin_col], train[tar_col], cv=cv_split)
    vote_soft.fit(train[feat_bin_col], train[tar_col])

    print("Soft Voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean() * 100))
    print("Soft Voting Test w/bin score mean: {:.2f}".format(vote_soft_cv['test_score'].mean() * 100))
    print("Soft Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_soft_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # ################# 2
    # Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]

    grid_param = [
        [{
            # AdaBoostClassifier
            'n_estimators': grid_n_estimator,  # default=50
            'learning_rate': grid_learn,  # default=1
            # 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
        }],

        [{
            # BaggingClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'max_samples': grid_ratio,  # default=1.0
            'random_state': grid_seed
        }],

        [{
            # ExtraTreesClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'criterion': grid_criterion,  # default=”gini”
            'max_depth': grid_max_depth,  # default=None
            'random_state': grid_seed
        }],

        [{
            # GradientBoostingClassifier
            #  'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': [.05],# default=0.1 -- 12/31/17 set to reduce runtime
            # -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': [300],# default=100 -- 12/31/17 set to reduce runtime
            # 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth,  # default=3
            'random_state': grid_seed
        }],

        [{
            # RandomForestClassifier
            'n_estimators': grid_n_estimator,  # default=10
            'criterion': grid_criterion,  # default=”gini”
            'max_depth': grid_max_depth,  # default=None
            'oob_score': [True],# default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
        }],

        [{
            # GaussianProcessClassifier
            'max_iter_predict': grid_n_estimator,  # default: 100
            'random_state': grid_seed
        }],

        [{
            # LogisticRegressionCV
            'fit_intercept': grid_bool,  # default: True
            # 'penalty': ['l1','l2'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # default: lbfgs
            'random_state': grid_seed
        }],

        [{
            # BernoulliNB
            'alpha': grid_ratio,  # default: 1.0
        }],

        # GaussianNB -
        [{}],

        [{
            # KNeighborsClassifier
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7],  # default: 5
            'weights': ['uniform', 'distance'],  # default = ‘uniform’
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }],

        [{
            # SVC
            # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1, 2, 3, 4, 5],  # default=1.0
            'gamma': grid_ratio,  # edfault: auto
            'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
            'probability': [True],
            'random_state': grid_seed
        }],

        [{
            # XGBClassifier
            'learning_rate': grid_learn,  # default: .3
            'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
            'n_estimators': grid_n_estimator,
            'seed': grid_seed
        }]
    ]

    start_total = time.perf_counter()
    for clf, param in zip(vote_est, grid_param):
        # print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
        # print(param)

        start = time.perf_counter()
        best_search = model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc')
        best_search.fit(train[feat_bin_col], train[tar_col])
        run = time.perf_counter() - start

        best_param = best_search.best_params_
        print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__,
                                                                                         best_param, run))
        clf[1].set_params(**best_param)

    run_total = time.perf_counter() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total / 60))

    print('-' * 10)

    # ################### 3.
    # Hard Vote or majority rules w/Tuned Hyperparameters
    grid_hard = ensemble.VotingClassifier(estimators=vote_est, voting='hard')
    grid_hard_cv = model_selection.cross_validate(grid_hard, train[feat_bin_col], train[tar_col], cv=cv_split)
    grid_hard.fit(train[feat_bin_col], train[tar_col])

    print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".format(
        grid_hard_cv['train_score'].mean() * 100))
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".format(
        grid_hard_cv['test_score'].mean() * 100))
    print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".format(
        grid_hard_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # Soft Vote or weighted probabilities w/Tuned Hyperparameters
    grid_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    grid_soft_cv = model_selection.cross_validate(grid_soft, train[feat_bin_col], train[tar_col], cv=cv_split)
    grid_soft.fit(train[feat_bin_col], train[tar_col])

    print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".format(
        grid_soft_cv['train_score'].mean() * 100))
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".format(
        grid_soft_cv['test_score'].mean() * 100))
    print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".format(
        grid_soft_cv['test_score'].std() * 100 * 3))
    print('-' * 10)

    # 12/31/17 tuned with data1_x_bin
    # The best parameter for AdaBoostClassifier is {'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0} with a runtime of 33.39 seconds.
    # The best parameter for BaggingClassifier is {'max_samples': 0.25, 'n_estimators': 300, 'random_state': 0} with a runtime of 30.28 seconds.
    # The best parameter for ExtraTreesClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0} with a runtime of 64.76 seconds.
    # The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 34.35 seconds.
    # The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 76.32 seconds.
    # The best parameter for GaussianProcessClassifier is {'max_iter_predict': 10, 'random_state': 0} with a runtime of 6.01 seconds.
    # The best parameter for LogisticRegressionCV is {'fit_intercept': True, 'random_state': 0, 'solver': 'liblinear'} with a runtime of 8.04 seconds.
    # The best parameter for BernoulliNB is {'alpha': 0.1} with a runtime of 0.19 seconds.
    # The best parameter for GaussianNB is {} with a runtime of 0.04 seconds.
    # The best parameter for KNeighborsClassifier is {'algorithm': 'brute', 'n_neighbors': 7, 'weights': 'uniform'} with a runtime of 4.84 seconds.
    # The best parameter for SVC is {'C': 2, 'decision_function_shape': 'ovo', 'gamma': 0.1, 'probability': True, 'random_state': 0} with a runtime of 29.39 seconds.
    # The best parameter for XGBClassifier is {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0} with a runtime of 46.23 seconds.
    # Total optimization time was 5.56 minutes.


    # ################# 4.
    # prepare data for modeling
    test=[]
    print(test.info())
    print("-" * 10)
    # data_val.sample(10)

    # handmade decision tree - submission score = 0.77990
    # test['Survived'] = mytree(test).astype(int)

    # decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
    # submit_dt = tree.DecisionTreeClassifier()
    # submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
    # submit_dt.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_dt.best_params_) #Best Parameters:  {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}
    # data_val['Survived'] = submit_dt.predict(data_val[data1_x_bin])

    # bagging w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77990
    # submit_bc = ensemble.BaggingClassifier()
    # submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
    # submit_bc.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_bc.best_params_) #Best Parameters:  {'max_samples': 0.25, 'n_estimators': 500, 'oob_score': True, 'random_state': 0}
    # data_val['Survived'] = submit_bc.predict(data_val[data1_x_bin])

    # extra tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
    # submit_etc = ensemble.ExtraTreesClassifier()
    # submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
    # submit_etc.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_etc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
    # data_val['Survived'] = submit_etc.predict(data_val[data1_x_bin])

    # random foreset w/full dataset modeling submission score: defaults= 0.71291, tuned= 0.73205
    # submit_rfc = ensemble.RandomForestClassifier()
    # submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
    # submit_rfc.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_rfc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
    # data_val['Survived'] = submit_rfc.predict(data_val[data1_x_bin])

    # ada boosting w/full dataset modeling submission score: defaults= 0.74162, tuned= 0.75119
    # submit_abc = ensemble.AdaBoostClassifier()
    # submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
    # submit_abc.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_abc.best_params_) #Best Parameters:  {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0}
    # data_val['Survived'] = submit_abc.predict(data_val[data1_x_bin])

    # gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033
    # submit_gbc = ensemble.GradientBoostingClassifier()
    # submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)
    # submit_gbc.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_gbc.best_params_) #Best Parameters:  {'learning_rate': 0.25, 'max_depth': 2, 'n_estimators': 50, 'random_state': 0}
    # data_val['Survived'] = submit_gbc.predict(data_val[data1_x_bin])

    # extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990
    # submit_xgb = XGBClassifier()
    # submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)
    # submit_xgb.fit(data1[data1_x_bin], data1[Target])
    # print('Best Parameters: ', submit_xgb.best_params_) #Best Parameters:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}
    # data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])

    # hard voting classifier w/full dataset modeling submission score: defaults= 0.75598, tuned = 0.77990
    # data_val['Survived'] = vote_hard.predict(data_val[data1_x_bin])
    test['Survived'] = grid_hard.predict(test[feat_bin_col])

    # soft voting classifier w/full dataset modeling submission score: defaults= 0.73684, tuned = 0.74162
    # data_val['Survived'] = vote_soft.predict(data_val[data1_x_bin])
    # data_val['Survived'] = grid_soft.predict(data_val[data1_x_bin])

    # submit file
    submit = test[['PassengerId', 'Survived']]
    submit.to_csv("../working/submit.csv", index=False)

    print('Validation Data Distribution: \n', test['Survived'].value_counts(normalize=True))
    submit.sample(10)



if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    combine = [train, test]

    # print train.info()
    # print '-'*10, 'train isnull'
    # print train.isnull().sum()
    # print '-'*10, 'test isnull'
    # print test.isnull().sum()
    # print train.describe(include=['O'])
    # print '-'*20
    # print train.describe(include='all')

    # #################### 1.change features
    clean_data(train, test)

    # ################### 2.Check infos
    # print train.isnull().sum()
    # print '-'*10
    # print test.isnull().sum()

    # print train.info()
    # print '-'*10
    # print test.info()

    # print train.head()

    # #################### 3.Convert Formats
    convert_formats(combine)

    # #################### 4.Define features
    tar_col = ['Survived']
    # 4.1 feature selection
    feat_sel_col = ['Sex', 'Pclass', 'Embarked', 'Title',
                   'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
    feat_cal_col = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code',
                    'SibSp', 'Parch', 'Age', 'Fare']
    tar_feat_sel_col = tar_col + feat_sel_col

    # 4.2 bin feature to remove continuous variables
    feat_bin_col = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code',
                    'FamilySize', 'AgeBin_Code', 'FareBin_Code']
    tar_feat_bin_col = tar_col + feat_bin_col

    # 4.3 dummy features
    data_dummy = pd.get_dummies(train[feat_sel_col])
    feat_dummy_col = data_dummy.columns.tolist()
    tar_feat_dummy_col = tar_col + feat_dummy_col
    # print '-'*20, 'dummy'
    # print data_dummy.head()

    # #################### 5. Double Check
    # print train.info()
    # print '-'*10
    # print test.info()

    # #################### 6. Split data
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
        train[feat_cal_col], train[tar_col], random_state=0)
    train_X_bin, test_X_bin, train_Y_bin, test_Y_bin = model_selection.train_test_split(
        train[feat_bin_col], train[tar_col], random_state=0)
    train_X_dummy, test_X_dummy, train_Y_dummy, test_Y_dummy = model_selection.train_test_split(
        data_dummy[feat_dummy_col], train[tar_col], random_state=0)
    print train_X_dummy.shape, test_Y_dummy.shape
    print '-'*20, 'train_X'
    print train_X.head()
    print '-'*20, 'train_X_bin'
    print train_X_bin.head()
    print '-'*20, 'train_X_dummy'
    print train_X_dummy.head()

    # ##################### 7. Explore data with descriptive and graphical statistics
    # explore_data(train,feat_sel_col,tar_col)

    # ##################### 8. Model, just have a try
    # try_model(train,tar_col,feat_bin_col)

    # ##################### 9. Tune Model with Hyper-Parameters
    # tune_dt(train,feat_bin_col,tar_col)

    # ##################### 10. Voting model
    # validate_implement(train,feat_bin_col,tar_col)

    plt.show()
