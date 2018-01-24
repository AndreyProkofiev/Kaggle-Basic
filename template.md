[TOC]

# 1. import

## 1.1 features

```python
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import random
import re
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
```

## 1.2 models

```python
# from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
#                               GradientBoostingClassifier, ExtraTreesClassifier)
# from sklearn.svm import SVC
# from sklearn.cross_validation import KFold

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, 
					discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
```





# 2. meet

```python
train = pd.read_csv('../input/train.csv')
# Read in the CSV file and convert "?" to NaN
#df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/autos/imports-85.data",header=None, names=headers, na_values="?" )

print(train_df.columns.values)
train.head(3)
train.sample(10)
print('Train columns with null values:\n', train.isnull().sum())
train['Sex'].value_counts()
train.dtypes
train[train.isnull().any(axis=1)] # any():whether any element is True over requested axis

# x entries,0-x-1 ; total x columns; xxx:x none-null int64; dtypes:category(2); memory usage
train.info()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe
# count unique top freq mean std min 25% 50% 75% max
data_raw.describe(include = 'all') # ['O'], ['category'], [numpy.number], None:numeric

# ########## groupby
print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))

# ########## crosstab
pd.crosstab(train['Title'], train['Sex'])
```



# 3. feature engineering

```python
full_data = [train, test]

# ########## apply
train['Name_length'] = train['Name'].apply(len)
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# ########## base, loc
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
# ########## dropna() mode()[0]
freq_port = train.Embarked.dropna().mode()[0]

# ########## fillna
for dataset in full_data:
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    
# ########## fill age with avg, std
# method 1
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
# method 2: by pclass and sex
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
    
# ########## cut, use .loc[ dataset['Fare'] <= 7.91, 'Fare']=1 and drop farebin
for dataset in full_data:
    dataset['FareBin'] = pd.qcut(train['Fare'], 4) # frequency: (31.0, 512.329]
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5) # equal bin width (16.0, 32.0]
    
# ########## title
# method 1
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# method 2
	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# method 3
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# ########## value_counts
title_names = (train['Title'].value_counts() < 10)
for dataset in full_data:
	dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
# ########## replace
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'], 'Rare')
    cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                	"num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,"two": 2, 										"twelve": 12, "three":3 }}
    dataset.replace(cleanup_nums, inplace=True)

# ########## mapping
for dataset in full_data:
    # discrete
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    # continous
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset['Fare'] = dataset['Fare'].astype(int)
    
# ########## categorical encoder
# method 1
label = LabelEncoder()
for dataset in full_data:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
# method 2
    dataset["body_style"] = dataset["body_style"].astype('category')
    dataset["body_style_cat"] = dataset["body_style"].cat.codes
# method 3: dummy
	dataset = pd.get_dummies(train['Sex']) # Sex_female	Sex_male
    
# ########## drop features
drop_column = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train.drop(drop_column, axis = 1, inplace = True)
train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1, inplace = True)
test.drop(drop_column, axis = 1, inplace = True)
```



# 4. plot

## sns: Heatmap

```python
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 	linecolor='white', annot=True)
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232245_46.png)

```python
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train)
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232320_154.png)



## sns: Pairplot

<https://www.jianshu.com/p/6e18d21a4cad>

```python
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked', u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2, diag_kind ='kde', diag_kws=dict(shade=True),plot_kws=dict(s=10))
g.set(xticklabels=[])

# pp = sns.pairplot(train, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
# pp.set(xticklabels=[])
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232251_591.png)

![](http://omqwn4oyr.bkt.clouddn.com/201801232319_54.png)



## sns: distplot 

```python
sns.set_style('dark')                # 该图使用黑色为背景色
sns.distplot(births['prglngth'], kde=False) # 不显示密度曲线
sns.axlabel('Birth number', 'Frequency') # 设置X轴和Y轴的坐标含义
sns.plt.show()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232354_570.png)



## plt: Boxplot & subplot

```python
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=train['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(train['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(train['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.show()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232256_575.png)



## sns: Barplot & subplot

```python
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=train, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=train, ax = saxis[0,2])
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232303_884.png)



## sns: Pointplot & subplot

```python
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.pointplot(x = 'FareBin', y = 'Survived',  data=train, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=train, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=train, ax = saxis[1,2])

# two lines in a plot
# sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train,palette={"male": "blue", "female": "pink"},markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232304_520.png)



## sns: Violinplot & boxplot

```python
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = train, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train, split = True, ax=axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = train, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232307_629.png)



## sns: FacetGrid & pointplot

```python
e = sns.FacetGrid(train, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232312_548.png)



## sns: FacetGrid & kdeplot

核密度

```python
a = sns.FacetGrid( train, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , train['Age'].max()))
a.add_legend()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232314_452.png)



## sns: FacetGrid & barplot

> 条形图中间有间隔  多用于分类数据作图
>
> 直方图各条中间没有间隔  多用于连续型数据分段作图

```python
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
```



![](http://omqwn4oyr.bkt.clouddn.com/201801232328_760.png)



## sns.FacetGrid & plt.hist

> 条形图中间有间隔  多用于分类数据作图
> 直方图各条中间没有间隔  多用于连续型数据分段作图

- 不同阶级、性别、年龄的生存情况

  ```python
  h = sns.FacetGrid(train, row = 'Sex', col = 'Pclass', hue = 'Survived')
  h.map(plt.hist, 'Age', alpha = .75)
  h.add_legend()
  ```

![](http://omqwn4oyr.bkt.clouddn.com/201801232316_960.png)

- 不同阶级、性别的年龄分布

  ```python
  grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', alpha=.5, bins=20)
  grid.add_legend()
  ```

  ![](http://omqwn4oyr.bkt.clouddn.com/201801232342_553.png)

  ​

## plt: Hist & subplot

```python
plt.figure(figsize=[16,12])

plt.subplot(234)
plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [train[train['Survived']==1]['FamilySize'], train[train['Survived']==0]['FamilySize']], stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

plt.show()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232259_937.png)





# 5. model

> ```
> # ensemble
> AdaBoostClassifier(), BaggingClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), RandomForestClassifier();
>
> # gaussian_process
> GaussianProcessClassifier();
>
> # linear_model
> LogisticRegressionCV(), PassiveAggressiveClassifier(), RidgeClassifierCV(), SGDClassifier(), Perceptron();
>
> # naive_bayes
> BernoulliNB(), GaussianNB();
>
> # neighbors
> KNeighborsClassifier();
>
> # svm
> SVC(probability=True), NuSVC(probability=True), LinearSVC();
>
> # tree    
> DecisionTreeClassifier(), ExtraTreeClassifier();
>
> # discriminant_analysis
> LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis();
>
> #xgboost
> XGBClassifier()   
> ```



## 5.0 Common Operators

### 5.0.1 Split 

```python
train: X_train, Y_train, X_val, Y_val, Y_pred
test: test, predictions
    
# ########## train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, 
                                                  random_state=random_seed)
# stratify = y, will let the train and test have the same proportion
# https://www.cnblogs.com/bettyty/p/6357760.html
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, 
                                                 random_state=random_seed,stratify=Y_train)

# ########## StratifiedKFold：data is not balanced, 每个类别的数量一样
# a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class. if shuffle=True, the data is shuffled once at the start. random_state used when shuffle=True
cv_split = model_selection.StratifiedKFold(n_splits=10, shuffle=False)

# ########## StratifiedShuffleSplit
# a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class. The data will overlap.
cv_split = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.25, 
                                                  random_state=0)

# https://stackoverflow.com/questions/45969390/difference-between-stratifiedkfold-and-stratifiedshufflesplit-in-sklearn
# ########## ShuffleSplit：有交集
# In ShuffleSplit, the data is shuffled every time, and then split. This means the test sets may overlap between the splits.
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, 
                                        random_state = 0 ) 

# ########## KFold：按顺序k-fold，不shuffle
# In KFolds, each test set should not overlap, even with shuffle. With KFolds and shuffle, the data is shuffled once at the start, and then divided into the number of desired splits. The test data is always one of the splits, the train data is the rest.
cv_split = model_selection.KFold(n_splits=10, shuffle=False)
```

### 5.0.1 Train

- fit without cv

  ```python
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, 
                                                    random_state=random_seed)

  clf = linear_model.LogisticRegressionCV()
  clf.fit(X_train, Y_train)
  print 'train accuary:',clf.score(X_train, Y_train)
  print 'val accuary:',clf.score(X_val, Y_val)
  ```

- fit with cv

  ```python
  lasso = linear_model.Lasso()
  cv_results = cross_validate(lasso, X, y, return_train_score=False, cv=cv_split)

  print('AFTER Parameters: ', cv_results.best_params_)
  print("AFTER Training set score mean: {:.2f}".
        format(cv_results['train_score'].mean()*100)) 
  print("AFTER Test set score mean: {:.2f}".
        format(cv_results['test_score'].mean()*100))
  print("AFTER Test set score min: {:.2f}".
        format(cv_results['test_score'].min()*100))
  print('-'*10)
  ```

- optimize hyper-parameter without cv

  ```python
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, 
                                                    random_state=random_seed)

  param_grid = [
      {'kernel': ['linear']}, 
  	{'kernel': ['rbf'], 'gamma': [1, 10]}
  ]
  # [{'kernel': 'linear'}, {'kernel': 'rbf', 'gamma': 1}, {'kernel': 'rbf', 'gamma': 10}]

  for param in list(ParameterGrid(param_grid)):
      print param
      clf = linear_model.LogisticRegressionCV()
      clf.set_params(**param)
      clf.fit(X_train, Y_train)
      print 'train accuary:',clf.score(X_train, Y_train)
      print 'val accuary:',clf.score(X_val, Y_val)
  ```

- optimize hyper-parameter with cv

  ```python
  param_grid = {
      'criterion': ['gini', 'entropy'], 
      'max_depth': [2,4,6,8,10,None],
      'random_state': [0] 
  }

  tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), cv = cv_split, 
                                            param_grid=param_grid, scoring = 'roc_auc')
  tune_model.fit(X, Y)

  # print(tune_model.cv_results_.keys())
  # print(tune_model.cv_results_['params'])
  print('After Parameters: ', tune_model.best_params_)
  # print(tune_model.cv_results_['mean_train_score'])
  print("After Training score mean: {:.2f}". format(
      tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
  # print(tune_model.cv_results_['mean_test_score'])
  print("After Test score mean: {:.2f}". format(
      tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
  print("After Test score 3*std: +/- {:.2f}".format(
      tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
  print('-'*10)
  ```

  cv_results:

  > ```
  > {
  >   'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
  >                                mask = [False False False False]...)
  >   'param_gamma': masked_array(data = [-- -- 0.1 0.2],
  >                               mask = [ True  True False False]...),
  >   'param_degree': masked_array(data = [2.0 3.0 -- --],
  >                                mask = [False False  True  True]...),
  >   'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
  >   'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
  >   'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
  >   'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
  >   'rank_test_score'    : [2, 4, 3, 1],
  >   'split0_train_score' : [0.8, 0.9, 0.7],
  >   'split1_train_score' : [0.82, 0.5, 0.7],
  >   'mean_train_score'   : [0.81, 0.7, 0.7],
  >   'std_train_score'    : [0.03, 0.03, 0.04],
  >   'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
  >   'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
  >   'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
  >   'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
  >   'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
  > }
  > ```

## 5.1. Base

<https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy>

### 5.1.1 training

```python
#Machine Learning Algorithm (MLA) Selection and Initialization
algs = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost
    XGBClassifier()    
]

# run model 10x with 60/30 split intentionally leaving out 10%
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, 
                                        random_state = 0 ) 

#create table to compare MLA metrics
columns = ['Name', 'Parameters','Train_Acc_Mean', 'Test_Acc_Mean', 'Test_Acc_3*STD' ,'Time']
compare = pd.DataFrame(columns = columns)

#create table to compare MLA predictions
predict = Y_train

#index through MLA and save performance to table
row_index = 0
for alg in algs:
    #set name and parameters
    name = alg.__class__.__name__
    compare.loc[row_index, columns[0]] = name
    compare.loc[row_index, columns[1]] = str(alg.get_params())
    
    #score model with cross validation
    cv_results = model_selection.cross_validate(alg, X_train, Y_train, 
                                                cv  = cv_split)

    compare.loc[row_index, columns[5]] = cv_results['fit_time'].mean()
    compare.loc[row_index, columns[2]] = cv_results['train_score'].mean()
    compare.loc[row_index, columns[3]] = cv_results['test_score'].mean()   
    compare.loc[row_index, columns[4]] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    
    #save MLA predictions - see section 6 for usage
    alg.fit(X_train, Y_train)
    predict[MLA_name] = alg.predict(X_train)
    
    row_index+=1

#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print compare
```



### 5.1.2 compare: barplot

```python
sns.barplot(x='Test_Acc_Mean', y = 'Name', data = compare, color = 'm')

plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232322_87.png)

### 5.1.3 Tune 

```python
# ########## Tune Model with Hyper-Parameters
# 1. base model
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, X_train, Y_train, 
                                              cv  = cv_split)
dtree.fit(X_train, Y_train)

print('Before Parameters: ', dtree.get_params())
print("Before Training score mean: {:.2f}".format(
    base_results['train_score'].mean()*100)) 
print("Before Test score mean: {:.2f}". format(
    base_results['test_score'].mean()*100))
print("Before Test score 3*std: +/- {:.2f}".format(
    base_results['test_score'].std()*100*3))
print("Before Test set score min: {:.2f}".format(
    base_results['test_score'].min()*100))
print('-'*10)

# 2. GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'], 
    # 'splitter': ['best', 'random'], 
    'max_depth': [2,4,6,8,10,None],
    # 'min_samples_split': [2,5,10,.03,.05], 
    # 'min_samples_leaf': [1,5,10,.03,.05], 
    # 'max_features': [None, 'auto'], 
    'random_state': [0] 
}

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, 
                                          scoring = 'roc_auc', cv = cv_split)
tune_model.fit(X_train, Y_train)

# print(tune_model.cv_results_.keys())
# print(tune_model.cv_results_['params'])
print('After Parameters: ', tune_model.best_params_)
# print(tune_model.cv_results_['mean_train_score'])
print("After Training score mean: {:.2f}". format(
    tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
# print(tune_model.cv_results_['mean_test_score'])
print("After Test score mean: {:.2f}". format(
    tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("After Test score 3*std: +/- {:.2f}".format(
    tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)


# ########## Tune Model with Feature Selection
# 1. base model
print('Before RFE Training Shape Old: ', X_train.shape) 
print('Before RFE Training Columns Old: ', X_train.columns.values)
print("Before RFE Training score mean: {:.2f}". 
      format(base_results['train_score'].mean()*100)) 
print("Before RFE Test score mean: {:.2f}". 
      format(base_results['test_score'].mean()*100))
print("Before RFE Test score 3*std: +/- {:.2f}". 
      format(base_results['test_score'].std()*100*3))
print('-'*10)

# 2. feature selection
dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(X_train, Y_train)

X_rfe = X_train.columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, train[X_rfe], Y_train, cv  = cv_split)

# print(dtree_rfe.grid_scores_)
print('After RFE Training Shape New: ', train[X_rfe].shape) 
print('After RFE Training Columns New: ', X_rfe)
print("After RFE Training score mean: {:.2f}". 
      format(rfe_results['train_score'].mean()*100)) 
print("After RFE Test score mean: {:.2f}". 
      format(rfe_results['test_score'].mean()*100))
print("After RFE Test score 3*std: +/- {:.2f}". 
      format(rfe_results['test_score'].std()*100*3))
print('-'*10)

# 3. select features and then tune rfe model
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), cv = cv_split, 
                                              param_grid=param_grid, scoring = 'roc_auc')
rfe_tune_model.fit(train[X_rfe], Y_train)

# print(rfe_tune_model.cv_results_.keys())
# print(rfe_tune_model.cv_results_['params'])
print('After RFE Tuned Parameters: ', rfe_tune_model.best_params_)
# print(rfe_tune_model.cv_results_['mean_train_score'])
print("After RFE Tuned Training score mean: {:.2f}". 
      format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
# print(rfe_tune_model.cv_results_['mean_test_score'])
print("After RFE Tuned Test score mean: {:.2f}". 
      format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("After RFE Tuned Test score 3*std: +/- {:.2f}". 
      format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)
```



### 5.1.4 GridSearchCV 

```python
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
	# ########## ensemble
	# AdaBoostClassifier
	[{
		'n_estimators': grid_n_estimator, #default=50
		'learning_rate': grid_learn, #default=1
		# 'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
		'random_state': grid_seed
	}],
	# BaggingClassifier
	[{
		'n_estimators': grid_n_estimator, #default=10
		'max_samples': grid_ratio, #default=1.0
		'random_state': grid_seed
	}],
	# ExtraTreesClassifier
	[{
		'n_estimators': grid_n_estimator, #default=10
		'criterion': grid_criterion, #default=”gini”
		'max_depth': grid_max_depth, #default=None
		'random_state': grid_seed
	}],
	# GradientBoostingClassifier
	[{
		# 'loss': ['deviance', 'exponential'], #default=’deviance’
		'learning_rate': [.05], #default=0.1
		'n_estimators': [300], #default=100
		# 'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
		'max_depth': grid_max_depth, #default=3   
		'random_state': grid_seed
	 }],
	# RandomForestClassifier
	[{
		'n_estimators': grid_n_estimator, #default=10
		'criterion': grid_criterion, #default=”gini”
		'max_depth': grid_max_depth, #default=None
		'oob_score': [True], #default=False
		'random_state': grid_seed
	}],

	# ########## gaussian_process
	# GaussianProcessClassifier
	[{    
		'max_iter_predict': grid_n_estimator, #default: 100
		'random_state': grid_seed
	}],

	# ########## linear_model
	# LogisticRegressionCV 
	[{
		'fit_intercept': grid_bool, #default: True
		#'penalty': ['l1','l2'],
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
		'random_state': grid_seed
	}],
	# PassiveAggressiveClassifier()
	# RidgeClassifierCV()
	# SGDClassifier()
	# Perceptron()

	# ########## naive_bayes
	# BernoulliNB 
	[{
		'alpha': grid_ratio, #default: 1.0
	}],
	# GaussianNB - 
	[{}],

	# ########## neighbors
	# KNeighborsClassifier 
	[{
		'n_neighbors': [1,2,3,4,5,6,7], #default: 5
		'weights': ['uniform', 'distance'], #default = ‘uniform’
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
	}],

	# ########## svm
	# SVC
	[{
		#'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
		'C': [1,2,3,4,5], #default=1.0
		'gamma': grid_ratio, #edfault: auto
		'decision_function_shape': ['ovo', 'ovr'], #default:ovr
		'probability': [True],
		'random_state': grid_seed
	}],
	# NuSVC()
	# LinearSVC()

	# ########## tree    
	# DecisionTreeClassifier()
	# ExtraTreeClassifier()

	# ########## discriminant_analysis
	# LinearDiscriminantAnalysis()
	# QuadraticDiscriminantAnalysis()

	# ########## xgboost
	# XGBClassifier
	[{
		'learning_rate': grid_learn, #default: .3
		'max_depth': [1,2,4,6,8,10], #default 2
		'n_estimators': grid_n_estimator, 
		'seed': grid_seed  
	}]   
]

algs = [
	# ########## ensemble
	('ada', ensemble.AdaBoostClassifier()),
	('bc', ensemble.BaggingClassifier()),
	('etc',ensemble.ExtraTreesClassifier()),
	('gbc', ensemble.GradientBoostingClassifier()),
	('rfc', ensemble.RandomForestClassifier()),

	# ########## gaussian_process
	('gpc', gaussian_process.GaussianProcessClassifier()),

	# ########## linear_model    
	('lr', linear_model.LogisticRegressionCV()),
    # PassiveAggressiveClassifier()
	# RidgeClassifierCV()
	# SGDClassifier()
	# Perceptron()

	# ########## naive_bayes    
	('bnb', naive_bayes.BernoulliNB()),
	('gnb', naive_bayes.GaussianNB()),

	# ########## neighbors
	('knn', neighbors.KNeighborsClassifier()),

	# ########## svm
	('svc', svm.SVC(probability=True)),
	# NuSVC()
	# LinearSVC()
    
	# ########## tree    
	# DecisionTreeClassifier()
	# ExtraTreeClassifier()

	# ########## discriminant_analysis
	# LinearDiscriminantAnalysis()
	# QuadraticDiscriminantAnalysis()
    
	# ########## xgboost
	('xgb', XGBClassifier())
]

start_total = time.perf_counter()
for clf, param in zip(algs, grid_param):
	start = time.perf_counter()        
	best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, 
		cv = cv_split, scoring = 'roc_auc')
	best_search.fit(X_train, Y_train)
	run = time.perf_counter() - start

	best_param = best_search.best_params_
	print('{} -- best_param: {} -- runtime: {:.2f} seconds.'.format(
		clf[1].__class__.__name__, best_param, run))
	clf[1].set_params(**best_param) 

run_total = time.perf_counter() - start_total
print('Total: {:.2f} minutes.'.format(run_total/60))
print('-'*10)
```



## 5.2 Ensembling model

<https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook>

### 5.2.1 Training 

```python
# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
        
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
# Class to extend XGboost classifer

# ###########
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
}

# ###########
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# ###########
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# ###########
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
```

###  5.2.2 Feature importance

```python
rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })
```

### 5.2.3 Second Level: XGBoost

```python
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
	#learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)
```



## Keras + TensorFlow

<https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6>

### import

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')
```

### feature

```python
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, 
                                                  random_state=random_seed)

g = plt.imshow(X_train[0][:,:,0])
```

### training

```python
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

# ######### construction
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# ########## training
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

# ########## predict
results = model.predict(test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

# ########## submission
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
```

### error curve

```python
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
```





# 6. Submit

```python
# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
```

