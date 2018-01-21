from pandas import Series,DataFrame
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

# s = Series(data=[1,2,3,4],index=['a','b','c', 'd'])
# print s
#
# data = {'state': ['Ohino','Ohino','Ohino','Nevada','Nevada'],
#         'year': [2000,2001,2002,2001,2002],
#         'pop': [1.5,1.7,3.6,2.4,2.9]}
# df = DataFrame(data)
# print df
# print df.drop('state',axis=1)
# print df.drop(0,axis=0)

# print '-'*20
data = pd.read_csv('test.csv')
print data

print type(data['class'][0])
print data.ix[:2, 2:3]
print data.ix[0,:]
print len(data.ix[0,:])

# print 'mode', data['class'].mode()[0]
# print pd.cut(data['class'].astype(int), 2)
# print pd.qcut(data['class'].astype(int), 2)
# print data['pop'].str.split(',', expand=True)[1]
# print data['pop'].str.split(',', expand=False)

print '-'*10

label = LabelEncoder()
# label.fit([100,0,1,2])
# data['Class_Code'] = label.transform(data['class'])
data['Sex_Code'] = label.fit_transform(data['sex'])
print pd.get_dummies(data)

group1 = data.groupby(['class','sex'])
for x in group1:
        print x

c = data['class']
c[0] = 10
print c
print '8'*20
print data


#########################
df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'prize', 'class label']
print df

size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1}
df['size'] = df['size'].map(size_mapping)

class_mapping = {label: idx for idx, label in enumerate(set(df['class label']))}
df['class label'] = df['class label'].map(class_mapping)

print df

print pd.get_dummies(df)
