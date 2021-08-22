"""
supervised learning - classification

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()

#to call data from iris

iris.feature_names

Data_iris = iris.data

# to change array to dataframe and to add column names 
Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)

#to add label into dataframes
Data_iris['label'] = iris.target

#to visualize data of iris
plt.scatter(Data_iris.iloc[:,2],Data_iris.iloc[:,3], c= iris.target)
plt.xlabel('Petal length(cm)')
plt.ylabel('Petal Width(cm)')
plt.show()

x = Data_iris.iloc[:,0:4]
y = Data_iris.iloc[:,4]


"""
k-NN Classifier

"""

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 1)

kNN.fit(x,y)

#predicting using sepal length and sepal width
x_new = np.array([[5.6,3.4,1.4,0.1]])
#predicting using new sample
kNN.predict(x_new)
#another sample
x_n2 = np.array([[7.5,4,5.5,2]])
kNN.predict(x_n2)

#train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8,
                                                    random_state = 88, shuffle=True,
                                                    stratify = y)

#random_state to return to same value using that number

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 1)

kNN.fit(X_train,y_train)

predicted_type = kNN.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predicted_type)



"""
Decision TREE

"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Dt = DecisionTreeClassifier()

Dt.fit(X_train, y_train)

predicted_dt_type = Dt.predict(X_test)

accuracy_score(y_test,predicted_dt_type)

#cross validation
from sklearn.model_selection import cross_val_score

#using x&y rather than x_test & y_test

Scores_dt = cross_val_score(Dt, x, y, cv=10)
#above gives accuracy for different test sets



"""
Naive Bayes Classifier

"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

NB = GaussianNB()

NB.fit(X_train,y_train)

Predicted_type_NB = NB.predict(X_test)

accuracy_score(y_test, Predicted_type_NB)

#cross validation
from sklearn.model_selection import cross_val_score

#using x&y rather than x_test & y_test

Scores_NB = cross_val_score(NB, x, y, cv=10)

"""
Logistic Regression

"""
from sklearn.datasets import load_breast_cancer

Data_c = load_breast_cancer()

x = Data_c.data
y = Data_c.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,train_size=0.7,
                                                  random_state=88)

from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(X_train, y_train)

predicted_classes_Lr = Lr.predict(X_test)

from sklearn.model_selection import cross_val_score
Scores_Lr = cross_val_score(Lr, x,y, cv=60)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predicted_classes_Lr)


"""
Evaluation Metrics

"""
from sklearn.metrics import confusion_matrix, classification_report

conf_mat = confusion_matrix(y_test,predicted_classes_Lr)

class_rep = classification_report(y_test,predicted_classes_Lr)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

y_prob = Lr.predict_proba(X_test)

y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)

plt.plot(FPR,TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_prob)


