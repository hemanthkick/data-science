# -*- coding: utf-8 -*-
"""
preprocessing

importing and modifying Dataset

"""

import pandas as pd

Data_set1 = pd.read_csv('Data_Set.csv')

Data_set2 = pd.read_csv('Data_Set.csv', header = 2)

Data_set3 = Data_set2.rename(columns = {'Temperature':'Temp'})

Data_set4 = Data_set3.drop('No. Occupants', axis = 1)

Data_set3.drop('No. Occupants', axis = 1, inplace = True)

Data_set5 = Data_set4.drop(2, axis = 0) #to remove row axis =0

Data_set6 = Data_set5.reset_index(drop = True) #if we do not write drop it keeps the old index

Data_set6.describe()

Min_item = Data_set6['E_Heat'].min()

Data_set6['E_Heat'][Data_set6['E_Heat'] == Min_item]

Data_set6['E_Heat'].replace(-4,21,inplace = True)
#inplace to implement our change in dataset

#check covariance

Data_set6.cov()

import seaborn as sns

sns.heatmap(Data_set6.corr())

"""""""""
Missing Values

"""

Data_set6.info()

#to find some other objects

import numpy as np

Data_set7 = Data_set6.replace('!',np.NaN)

Data_set7.info()

Data_set7 = Data_set7.apply(pd.to_numeric) #to change categorical to numeric

#to locate nan

Data_set7.isnull()

Data_set7.drop(13, axis=0, inplace = True)
# by running above we will delete 13 row

#to drop all nan values in one line code
Data_set7.dropna(axis=0, inplace = True)
#we lost whole rows
#other ways is to replace nan
#to go back we ran dataset7 again

Data_set8 = Data_set7.fillna(method = 'ffill')
#ffill for last observation, bfill for next observation to fill 

#to replace missing values with average/median/most frequent data
from sklearn.impute import SimpleImputer
M_var = SimpleImputer(missing_values= np.nan, strategy = 'mean')

M_var.fit(Data_set7)

Data_set9 = M_var.transform(Data_set7)

#outlier detection

Data_set8.boxplot()

Data_set8['E_Plug'].quantile(0.25)
Data_set8['E_Plug'].quantile(0.75)

#Q1= 19.75 & Q3= 32.25
#IQR = 32.25 - 19.75 = 12.5

# mild outlier
#lower Bound = Q1-1.5*IQR = 19.75 - 1.5*12.5 =1
#Upper Bound = Q3 + 1.5*IQR = 32.25 + (1.5*12.5) = 51

#extreme outlier
#upper Bound = Q3 +3*IQR = 32.25 + 3*12.5 = 69.75

#replacing the outlier
Data_set8['E_Plug'].replace(120,42,inplace=True)

"""
Concatenation

"""
New_col = pd.read_csv('Data_New.csv')

Data_set10 = pd.concat([Data_set8,New_col],axis=1)

Data_set10.drop(22, axis=0, inplace = True)

Data_set10.drop(13, axis=0, inplace = True)

Data_set10 = Data_set10.reset_index(drop = True)


"""

DUMMY VARIABLE

"""

Data_set10.info()

Data_set11 = pd.get_dummies(Data_set10)

Data_set11.info()

#normalization
from sklearn.preprocessing import minmax_scale, normalize

#first method : minmax_scale

Data_set12 = minmax_scale(Data_set11, feature_range=(0,1))

#second method : normalize

Data_set13 = normalize(Data_set12, norm='l2', axis = 0)
#axis = 0 for normalizing feature, axis=1 for normalizing each sample

Data_set13 = pd.DataFrame(Data_set13,columns=['time','E_Plug','E_heat','Price','Temp',
                                              'OffPeak','Peak'])




