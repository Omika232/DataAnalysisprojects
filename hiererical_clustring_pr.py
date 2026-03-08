#import pandas as pd

#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.cluster import AgglomerativeClustering
#df=pd.read_csv(r'Mall_Customers_SortData.csv')
#print(df)
#import scipy.cluster.hierarchy as sch
#dendrom = sch.dendrogram(sch.linkage(df, method = 'ward'))
#plt.show()
#model = AgglomerativeClustering(n_clusters=5)
#graph = model.fit_predict(df)
#df.to_csv()
#df['graph']=graph
#print(df)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


df = pd.read_csv(r'9pm_LogisticReg/Facebook_Ads_2.csv', encoding='latin1')
print(df)
sns.countplot(x= df['Country'],data=df,hue='Clicked')
plt.show()

#sns.countplot(x='Time Spent on Site',y='Clicked',data=df,hue='Clicked')
#plt.show()

#sns.countplot(x='Salary',y='Clicked',data=df,hue='Clicked')
#plt.show()

X = df.drop('Clicked',axis=1)
Y = df['Clicked']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=2)
model = LogisticRegression()
model.fit(X_train,Y_train)
Model = model.predict(X_test)
print("accuracy is : {accuracy_score(Model,Y_test}")