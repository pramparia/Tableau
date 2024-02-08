import numpy as np
import matplotlib as plt
import seaborn as sn
import sklearn as sk
import pandas as pd

fname=r'C:\Pooja\Masters/Global Superstore Orders 2016.xlsx'
mydata=pd.read_excel(fname, 'Orders')
mydata.head()
mydata.info()
mydata.describe()
