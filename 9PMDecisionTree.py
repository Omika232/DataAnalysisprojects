import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\ConsoleFlare_Classes\DecisionTree and Forest\kyphosis.csv")

# print(df)
# sns.countplot(data = df, x = "Age", hue  = "Kyphosis")
# plt.show()

# print(df)
# sns.countplot(data = df, x = "Number", hue  = "Kyphosis")
# plt.show()

# print(df)
# sns.countplot(data = df, x = "Start", hue  = "Kyphosis")
# plt.show()

X = df.drop("Kyphosis", axis = 1)
y = df["Kyphosis"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

decision = DecisionTreeClassifier()

decision.fit(X_train,y_train)

y_predict =  decision.predict(X_test)

print(accuracy_score(y_predict,y_test))

randomForestClassifier = RandomForestClassifier(7)

randomForestClassifier.fit(X_train,y_train)

y_predict =  randomForestClassifier.predict(X_test)

print(accuracy_score(y_predict,y_test))
