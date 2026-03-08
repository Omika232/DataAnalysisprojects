import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(r"kyphosis.csv")
print(df)
#sns.countplot(data=df, X='age', hue="Kyphosis")
#plt.show()
#sns.countplot(data=df, x="Number", hue= "Start" )
#plt.show()
#(sns.countplot(data=df,X = 'Start' , hue= 'Kyphosis')
#plt.show()
X = df.drop(labels="Kyphosis",axis=1)
Y=df["Kyphosis"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
decision = DecisionTreeClassifier()
decision.fit(X_train,Y_train)
Y_predict = decision.predict(X_test)
print(accuracy_score(Y_predict,Y_test))
Model = RandomForestClassifier()
Model.fit(X_train,Y_train)
Y_predict1=Model.predict(X_test)
print(accuracy_score(Y_predict1,Y_test))