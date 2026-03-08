import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error
df= pd.read_csv(r"9AM_Linear_Regression/IceCreamData.csv")
print(df)
sns.jointplot(dat= df, X='Temprature',Y='Revenue')
plt.show()

X= df.drop(["Revenue"],axis=1)
Y= df['Revenue']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
linear= LinearRegression()
linear.fit(X_train,Y_train)
Y_predict = linear.predict(X_test)
print(Y_predict)
print(Y_test)
accuracy = mean_squared_error(Y_test,Y_predict)
print("accuracy is ::")
print(math.sqrt(accuracy))
print("prdiction is ::")
print(linear.predict([[20]]))


#Practice 2


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math
df=pd.read_csv(r"9AM_Linear_Regression/FuelEconomy.csv")
print(df)

sns.catplot(data=df,x= 'Horse Power', y='Fuel Economy (MPG)')
plt.show()

X = df.drop(labels=["Fuel Economy (MPG)"],axis=1)
Y = df["Fuel Economy (MPG)"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
model= LinearRegression()
model.fit(X_train,Y_train)
predict = model.predict(X_test)
print(predict)
print(Y_test)
accuracy = mean_absolute_error(Y_test,predict)
print(accuracy)
print(math.sqrt(accuracy))
print(predict)
print(model.predict([[ 200]])) #this is two dimention so we use double brackets
