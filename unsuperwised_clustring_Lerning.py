import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df= pd.read_csv(r"Mall_Customers_SortData.csv")
print(df)
sns.scatterplot(data=df,x=df['Annual Income (k$)'],y=df['Spending Score (1-100)'])
plt.show()
l=[]
for i in range (1,10):
    model=KMeans(n_clusters=i)
    model.fit(df)
    l.append(model.inertia_)
print(l)
model=KMeans(n_clusters=5)
model.fit(df)
groups = model.predict(df)
df['group']=groups
#df.to_csv("")
print(model.predict([[25,30]]))
