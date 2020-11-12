# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:04:03 2020

@author: Kaan
"""
#Importing necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Importing sklearn algorithms
from sklearn.ensemble import RandomForestClassifier


#Creating the dataframe
main_df = pd.read_csv('term-deposit-marketing-2020.csv')

#Analyze the data for any null values, data types etc.
main_df.info() #No null values, 5 int and 9 object data types.

#Review the data statistically
main_df.describe()

#The age range of the costumer portfolio is between 19 and 95 with the majority
#between 30 and 50


main_df['job'].value_counts()
main_df['marital'].value_counts()
main_df['education'].value_counts()
main_df['month'].value_counts()


#Getting the data ready / Preprocessing

#Drop the contact data since it does not contribute here
main_df = main_df.drop(['contact'], axis=1)

(main_df['job'] == 'student').value_counts() #524 students out of the 40000
((main_df['job'] == 'student') & (main_df['y'] == 'yes')).value_counts()
#Only 82 student subscribed to the term deposit

#Binarizing the marital, education, default, housing, loan and y columns
dataset = [main_df]

marital_cat = {'single':0, 'married':1, 'divorced':2}
for data in dataset:
    data['marital'] = data['marital'].map(marital_cat)
    
    
education_cat = {'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3}
for data in dataset:
    data['education'] = data['education'].map(education_cat)
    

default_bin = {'no':0, 'yes':1}
for data in dataset:
    data['default'] = data['default'].map(default_bin)
    
    
housing_bin = {'no':0, 'yes':1}
for data in dataset:
    data['housing'] = data['housing'].map(housing_bin)
    
    
loan_bin = {'no':0, 'yes':1}
for data in dataset:
    data['loan'] = data['loan'].map(loan_bin)

y_bin = {'no':0, 'yes':1}
for data in dataset:
    data['y'] = data['y'].map(y_bin)

#Label encoding the job and month columns. Onehot encoding is
#not possible since there are too many categories

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

main_df['job'] = label_encoder.fit_transform(main_df['job'])

main_df['month'] = label_encoder.fit_transform(main_df['month'])



#Preparing the data for the algorithm. Train/Test split
from sklearn.model_selection import train_test_split


x = main_df.iloc[:,1:12].values
y = main_df.iloc[:,12:].values

x_train, x_test, y_train , y_test = train_test_split(x, y, test_size =0.2, random_state = 0)


#RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(x_train, y_train)
Y_pred = rfc.predict(x_test)
rfc.score(x_train, y_train)

#Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rfc, x_train, y_train, cv=5, scoring='accuracy')

#Scores after the cross validation
print('Scores:', scores)
print('Mean:', scores.mean())
print('Standard Deviation:', scores.std())


#Computing each feature's importance for random forest
importances = rfc.feature_importances_

for imp, val in enumerate(importances):
    print('Feature: %0d, Score: %.2f' % (imp, val))


plt.bar([x for x in range(len(importances))], importances)
plt.show()

#The features that impact whether the customer buys or not are
#the day feature, housing feature and the default feature
#Whether the customer already has a credit or not
