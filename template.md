- import 

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

- meet

  ```python
  train = pd.read_csv('../input/train.csv')
  print(train_df.columns.values)
  train.head(3)
  train.sample(10)
  print('Train columns with null values:\n', train.isnull().sum())
  train['Sex'].value_counts()

  # x entries,0-x-1 ; total x columns; xxx:x none-null int64; dtypes:category(2); memory usage
  train.info()
  # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe
  # count unique top freq mean std min 25% 50% 75% max
  data_raw.describe(include = 'all') # ['O'], ['category'], [numpy.number], None:numeric
  ```

- feature engineering

  ```python
  full_data = [train, test]

  # ########## apply
  train['Name_length'] = train['Name'].apply(len)
  train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

  # ########## base, loc
  for dataset in full_data:
      dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
      dataset['IsAlone'] = 0
      dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

  # ########## fillna
  for dataset in full_data:
      dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
      dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
      
  # ########## cut
  for dataset in full_data:
      dataset['FareBin'] = pd.qcut(train['Fare'], 4) # frequency: (31.0, 512.329]
      dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5) # equal bin width (16.0, 32.0]
      
  # ########## fill age with avg, std
  for dataset in full_data:
      age_avg = dataset['Age'].mean()
      age_std = dataset['Age'].std()
      age_null_count = dataset['Age'].isnull().sum()
      age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
      dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
      dataset['Age'] = dataset['Age'].astype(int)
      
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
  for dataset in full_data:
  	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

  # ########## value_counts
  title_names = (train['Title'].value_counts() < 10)
  for dataset in full_data:
  	dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
      
  # ########## replace
  for dataset in full_data:
      dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'], 'Rare')

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
  label = LabelEncoder()
  for dataset in full_data:    
      dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
      
  # ########## dummy
  train_dummy = pd.get_dummies(train['Sex']) # Sex_female	Sex_male

  #
  #
      
  # ########## drop features
  drop_column = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
  train.drop(drop_column, axis = 1, inplace = True)
  train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1, inplace = True)
  test.drop(drop_column, axis = 1, inplace = True)
  ```

- plot

- model