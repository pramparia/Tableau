# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:49:17 2023

@author: pooja
"""

import pandas as pd
import seaborn as sn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

credit_data = pd.read_csv('C:\Pooja\Masters/creditcard.csv')

y=credit_data.loc[:,['Class']]
x=credit_data.loc[:,~credit_data.columns.isin(['Class'])]

reg = LinearRegression()
reg.fit(x,y)
predic_class=reg.predict(x)

credit_data['Pred_reg']=predic_class
dis_score_fraud = credit_data.loc[credit_data['Class']==1,'Pred_reg'].mean()
dis_score_good = credit_data.loc[credit_data['Class']==0,'Pred_reg'].mean()
dis_score = 0.5*(dis_score_good + dis_score_fraud)

Pred_class = []

for i in range(len(y)):
    if credit_data['Pred_reg'][i]>=dis_score :
        Pred_class.append(1)
    else:
        Pred_class.append(0)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y,Pred_class)

n,m = conf_matrix.shape
conf_mat_rel = np.zeros((n, m))

for i in range(n):
    for j in range(m):
        conf_mat_rel[i][j]=np.round(conf_matrix[i][j]/sum(conf_matrix[i]),5)

conf_cm = pd.DataFrame(conf_mat_rel, range(2))
fig = plt.figure()
sn.heatmap(conf_cm, annot=True,fmt=".2%")
plt.show()
fig.savefig('confusion heatmap.pdf')