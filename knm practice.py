import  pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math

df=pd.read_csv(r"9PM_KNN/Iris.csv")
print(df)

#sns.countplot(x='SepalLengthCm',hue='Species',data=df)
#plt.show()

#sns.countplot(x='PetalLengthCm',hue='Species',data=df)
#plt.show()

#sns.countplot(x='PetalWidthCm',hue='Species',data=df)
#plt.show()

#sns.countplot(x='SepalWidthCm',hue='Species',data=df)
# plt.show()

X= df.drop('Species',axis=1)
Y = df['Species']
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,Y_train)
prediction_m = model.predict(X_test)
print("**prediction accuracy***")
print(accuracy_score(prediction_m,Y_test))
print(model.predict([[2,3,1.4,0.6]]))
