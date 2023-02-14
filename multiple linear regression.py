#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
from sklearn.linear_model import LinearRegression
x = [ [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35] ]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
print(x)
print(y)
model=LinearRegression().fit(x,y)
r2=model.score(x,y)
print(f"coeff:{r2}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print(f"predicted response:\n{y_pred}")
x_new = np.arange(10).reshape((-1, 2))
print(x_new)
y_new = model.predict(x_new)
print(y_new)


# In[28]:


import numpy as np #polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x= np.array([5,15,25,35,45,55]).reshape((-1,1)) #array 2D ie 1 col, many rows
y= np.array([5,20,14,32,22,38]) 
t=PolynomialFeatures(degree=2,include_bias=False)
t.fit(x)
a=t.transform(x)
print(a)
m=LinearRegression().fit(a,y)
r2=model.score(a,y)
print(f"coeff:{r2}")
print(f"intercept:{m.intercept_}")
print(f"coefficients: {model.coef_}")

