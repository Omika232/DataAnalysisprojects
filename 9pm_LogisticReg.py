import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("C:/ConsoleFlare_Classes/Logistic Regression/Train_Titanic.csv")

# sns.countplot(x = 'Pclass', hue= "Survived",data=df)
# plt.show()

# sns.countplot(x = 'Gender', hue= "Survived",data=df)
# plt.show()

# sns.countplot(x = 'SibSp', hue= "Survived",data=df)
# plt.show()

# sns.countplot(x = 'Parch', hue= "Survived",data=df)
# plt.show()

# sns.countplot(x = 'Age', hue= "Survived",data=df)
# plt.show()

# sns.countplot(x = 'Fare', hue= "Survived",data=df)
# plt.show()

df = df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

print(df.columns)

df["Age"]=df['Age'].fillna((df['Age'].mean()))

print(df)

label = LabelEncoder()
df["Gender"] = label.fit_transform(df["Gender"])

print(df)


X =  df.drop("Survived", axis = 1)
y = df["Survived"]

X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2, random_state=0)

logistic= LogisticRegression()

logistic.fit(X_train,y_train)

y_predict = logistic.predict(X_test)

print("accuracy is ::" )
print(accuracy_score(y_predict,y_test))





print(logistic.predict([[3,1,90,0,0,1]]))
