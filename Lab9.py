# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:59:34 2023

@author: pooja
"""

import pandas as pd
import seaborn as sn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_csv('C:\Pooja\Masters/Real estate.csv')

x=data.iloc[:,1:7]
y=data.iloc[:,7]

sn.pairplot(pd.concat([x,y],axis=1))

reg = LinearRegression()

reg.fit(x,y)

print("b0 = ", reg.intercept_)

print("b1 = ", reg.coef_)

print("R2 = ", reg.score(x,y))

model = sm.OLS(y,x).fit()
model.summary()
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
