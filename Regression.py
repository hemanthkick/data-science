"""
Regression

"""
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_p = load_boston()
x = Boston_p.data
y = Boston_p.target
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, train_size=0.75,
                                                    random_state=76)

#normalizing, i.e to make them into same range, and preprocessing
from sklearn.preprocessing import MinMaxScaler
Sc =  MinMaxScaler(feature_range = (0,1))

X_train = Sc.fit_transform(X_train)
X_test = Sc.fit_transform(X_test)

y_train = y_train.reshape(-1,1) #this is needed to avoid error
y_train = Sc.fit_transform(y_train)


"""
Multiple Linear Regression

"""
from sklearn.linear_model import LinearRegression

Linear_r = LinearRegression()

Linear_r.fit(X_train,y_train)

Predicted_values_mlr = Linear_r.predict(X_test)

# we get normalized values ie b/w 0&1
Predicted_values_mlr = Sc.inverse_transform(Predicted_values_mlr)


"""
Evaluation Metrics

"""
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

Mae = mean_absolute_error(y_test, Predicted_values_mlr)
Mse = mean_squared_error(y_test, Predicted_values_mlr)
Rmse = math.sqrt(Mse)

r2 = r2_score(y_test, Predicted_values_mlr)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

Mape = mean_absolute_percentage_error(y_test, Predicted_values_mlr)

"""
Polynomial linear regression

"""
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_p = load_boston()

x = Boston_p.data[:,5]
y = Boston_p.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, train_size=0.75,
                                                    random_state=76)

from sklearn.preprocessing import PolynomialFeatures

Poly_p = PolynomialFeatures(degree=2)
#for one dimension
X_train = X_train.reshape(-1,1)

poly_x = Poly_p.fit_transform(X_train)

from sklearn.linear_model import LinearRegression

Linear_r = LinearRegression()

Poly_L_R = Linear_r.fit(poly_x,y_train)

X_test = X_test.reshape(-1,1)

poly_xt = Poly_p.fit_transform(X_test)

Predicted_value_p = Poly_L_R.predict(poly_xt)

from sklearn.metrics import r2_score
r2 = r2_score(y_test,Predicted_value_p)


"""
Random Forest

"""

from sklearn.ensemble import RandomForestRegressor

Random_f = RandomForestRegressor(random_state=33)

Random_f.fit(X_train,y_train)

Predicted_val_Rf = Random_f.predict(X_test)

Predicted_val_Rf = Predicted_val_Rf.reshape(-1,1) #to un normalize

Predicted_val_Rf = Sc.inverse_transform(Predicted_val_Rf)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

Mae = mean_absolute_error(y_test, Predicted_val_Rf)
Mse = mean_squared_error(y_test, Predicted_val_Rf)
Rmse = math.sqrt(Mse)

r2 = r2_score(y_test, Predicted_val_Rf)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

Mape = mean_absolute_percentage_error(y_test, Predicted_val_Rf)

""" 
Support Vector Regression

"""

from sklearn.svm import SVR
#svm: support vector machine

Regressor_Svr = SVR(kernel='rbf')

Regressor_Svr.fit(X_train,y_train)

predicted_values_Svr = Regressor_Svr.predict(X_test)

predicted_values_Svr = predicted_values_Svr.reshape(-1,1)

predicted_values_Svr = Sc.inverse_transform(predicted_values_Svr)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

Mae = mean_absolute_error(y_test, predicted_values_Svr)
Mse = mean_squared_error(y_test, predicted_values_Svr)
Rmse = math.sqrt(Mse)

r2 = r2_score(y_test, predicted_values_Svr)

def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

Mape = mean_absolute_percentage_error(y_test, predicted_values_Svr)

















