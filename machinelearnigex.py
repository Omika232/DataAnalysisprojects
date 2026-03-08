import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sc
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

df= pd.read_csv(r'C:\Users\HP\Documents\Salary_Data.csv')
#print(df)
sc.relplot(x='YearsExperience',y='Salary',data=df)
#plt.show()

x=df.drop(['Salary'],axis=1)
y=df['Salary']

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.2,random_state=1)
print('*****training****')
print(x_train)
print(y_train)
print('***testing***')
print(x_test)
print(y_test)

model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print('**predictvalue**')
print(y_predict)
print('**actulvalue**')
print(y_test)

accuracy = mean_squared_error(y_test,y_predict)
print("accuracy is**")
print(math.sqrt(accuracy))
print(model.predict([[20]]))

