# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:47:59 2023

@author: pooja
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fname=r'C:\Pooja\Masters/Global Superstore Orders 2016.xlsx'

my_data = pd.read_excel(fname, 'Orders')

var_names = list(my_data)

x = my_data['Sales'].values
y = my_data['Profit'].values

plt.scatter(x, y)

reg = LinearRegression()
x = x.reshape(-1,1)
y = y.reshape(-1,1)

reg.fit(x,y)

reg.intercept_

reg.coef_

reg.score(x,y)
