import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('C:\Pooja\Masters/Real estate.csv')

x = data.iloc[:,1:7]# we're using the 2nd, 3rd and 4th fields as predictors
y = data.iloc[:,7] # we're using the last field (sales) are our dependent variable

n_simul = 1000 # number of simulations runs I want to perform
reg = LinearRegression()
R_squared_train = []
R_squared_test = []

for i in range(n_simul):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) # random splitting of the dataset
    reg.fit(x_train,y_train) # fitting the regression model
    R_squared_train.append(reg.score(x_train,y_train)) # retrieving the R squared
    y_pred = reg.predict(x_test) # predicting the y_test using the test predictors
# Next we calculate the out-of-sample R squared using the formula R^2 = 1 - SSE/SST
    SST_pred = sum((y_test - np.mean(y_test))**2)
    SSE_pred = sum((y_test-y_pred)**2)
    R_squared_pred = 1-SSE_pred/SST_pred
    R_squared_test.append(R_squared_pred)

R_squared = pd.DataFrame({'Training':R_squared_train,'Testing':R_squared_test})
stats = R_squared.describe()
print(stats)

plt.hist(R_squared_train)
plt.hist(R_squared_test)