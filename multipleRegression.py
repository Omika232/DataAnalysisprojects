import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df=pd.read_csv(r"Admission.csv")
print(df)
df= df.drop("Research",axis=1)
print(df)

#print(df.corr())
sns.heatmap(df.corr(),annot=True)
plt.show()

X =  df.drop(labels=['Admission Chance'],axis=1)
Y = df['Admission Chance']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
Model_1 = LinearRegression()
Model_1.fit(X_train,Y_train)
Y_predict = Model_1.predict(X_test)
print("*****error is******")
print(mean_absolute_error(Y_predict,Y_test))
print("*****accurcy is ***")
#print(math.sqrt())
print(Model_1.predict(([[332,118,7,4.5,4.5,9.66]])))

