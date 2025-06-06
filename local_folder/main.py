import pandas as pd
import numpy as np
df= pd.read_csv("C:\\Users\\H\\Downloads\\insurance.csv")
df['charges']=df['charges'].astype(int)
df['bmi']=df['bmi'].astype(int)
print(df.head())

from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['region'] = lb.fit_transform(df['region'])
print(df.head())

x = df.drop(columns = ['charges'])   
y = df['charges']   
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)

#for checking accuracy of the model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor()
rfr.fit(x_train,y_train)
Y_pred=rfr.predict(x_test)
print(r2_score(y_test,y_pred))
