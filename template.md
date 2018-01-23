[TOC]

# 1. import

```python
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
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

## 4.1 sns: Heatmap

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

correlation_heatmap(data1)
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232320_154.png)



## sns: Pairplot

<https://www.jianshu.com/p/6e18d21a4cad>

```python
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked', u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2, diag_kind ='kde', diag_kws=dict(shade=True),plot_kws=dict(s=10))
g.set(xticklabels=[])

# pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
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
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.show()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232256_575.png)



## sns: Barplot & subplot

```python
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232303_884.png)



## sns: Pointplot & subplot

```python
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])

# two lines in a plot
# sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,palette={"male": "blue", "female": "pink"},markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232304_520.png)



## sns: Violinplot & boxplot

```python
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax=axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232307_629.png)



## sns: FacetGrid & pointplot

```python
e = sns.FacetGrid(data1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232312_548.png)



## sns: FacetGrid & kdeplot

核密度

```python
a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
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
  h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
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

## 4.3 plt: Hist & subplot

```python
plt.figure(figsize=[16,12])

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

plt.show()
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232259_937.png)





# 5. model

## compare: barplot

```python
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
```

![](http://omqwn4oyr.bkt.clouddn.com/201801232322_87.png)

## Ensembling model

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
        print(self.clf.fit(x,y).feature_importances_)
        
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



# Submit

```python
# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
```

