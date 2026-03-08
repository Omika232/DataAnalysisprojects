import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("C:/ConsoleFlare_Classes/KMeanClustering/Mall_Customers_SortData.csv")

# sns.jointplot(x="Annual Income (k$)", y= "Spending Score (1-100)",data=df)
# plt.show()

distance =[]

for i in range(1,10):
    kmean = KMeans(n_clusters=i)
    kmean.fit(df)
    distance.append(kmean.inertia_)

kmean = KMeans(n_clusters=5)
kmean.fit(df)

y_predict = kmean.predict(df)

print(y_predict)

df["output"] = y_predict

sns.jointplot(x="Annual Income (k$)", y= "Spending Score (1-100)",data=df ,hue = "output",palette=["C0","C1","C2","C3","k"])
plt.show()


print(kmean.predict([[8,10]]))

print(df)