import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("C:/ConsoleFlare_Classes/KNN Algo/Iris.csv")

print(df)
#
# sns.countplot(x="SepalLengthCm",hue="Species",data=df)
# plt.show()

# sns.countplot(x="SepalWidthCm",hue="Species",data=df)
# plt.show()

# sns.countplot(x="PetalLengthCm",hue="Species",data=df)
# plt.show()

# sns.countplot(x="PetalWidthCm",hue="Species",data=df)
# plt.show()


X = df.drop("Species", axis=1)
y = df["Species"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

y_predict = knn.predict(X_test)

print(accuracy_score(y_predict,y_test))

print(knn.predict([[5.1,3.5,1.4,0.2]]))


joblib.dump(knn,r"C:\ConsoleFlare_Classes\KNN Algo\KNN.pkl")








