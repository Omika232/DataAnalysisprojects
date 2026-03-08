import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import math
df=pd.read_csv(r"9PM_KNN/Tshirt_Sizing_Dataset.csv")

#sns.countplot(data=df,x="Height (in cms)", hue= "T Shirt Size")
#plt.show()
#sns.countplot(data=df,x="Weight (in kgs)", hue= "T Shirt Size")
#plt.show()
X = df.drop('T Shirt Size',axis=1)
Y = df['T Shirt Size']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=9)
#model=KNeighborsClassifier(n_neighbors=5)
model=DecisionTreeClassifier()
#model=RandomForestClassifier()
model.fit(X_train,Y_train)
m_predict = model.predict(X_test)
print('**accuracy is**')
print(accuracy_score(m_predict,Y_test))
"*decision tree give 100% accuracy for this deta set so we use this"
