import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('vTargetMailCustomer.csv',encoding = 'unicode_escape')

TargetMailList = dataset[['CustomerKey','FirstName','MiddleName','LastName','MaritalStatus','Gender','EmailAddress','YearlyIncome','TotalChildren','NumberChildrenAtHome',	
                          'HouseOwnerFlag','NumberCarsOwned','CommuteDistance','Age','BikeBuyer']]

df = dataset[['MaritalStatus','Gender','YearlyIncome','NumberChildrenAtHome',	
                          'HouseOwnerFlag','NumberCarsOwned','CommuteDistance','Age','Region','BikeBuyer']]


cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
print(df['Region'].unique())
d1=pd.get_dummies(df.Region)
d2=pd.get_dummies(df.MaritalStatus)
df=pd.concat([df,d1,d2],axis=1)
df=df.rename(columns={'M':'Married','S':'Single'})
d3=pd.get_dummies(df.Gender)
df=pd.concat([df,d3],axis=1)
df=df.drop(['MaritalStatus','Gender','Region'],axis=1)
df['YearlyIncome']=df['YearlyIncome']/10000

print(df.CommuteDistance.unique())
df['CommuteDistance']=df['CommuteDistance'].apply(lambda x:'10-11' if x in ["10+ Miles"] else x)
df['CommuteDistance']=df['CommuteDistance'].apply(lambda x:int(x.split('-')[0]))
print(df['CommuteDistance'])

count=df.groupby('Age')['Age'].agg('count')
c=count[count<20]
df['Age']=df['Age'].apply(lambda x:90 if x in c else x)
print(df['Age'].unique())

X = df.iloc[:, :-1].values    
y = df.iloc[:, -1].values

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=.2)
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))
ypred=model.predict(xtest)
print(len(ytest))
ax=[0 for i in range(3697)]
for i in range(3697):
    ax[i]=i
t=[0.5 for i in range(3697)]
plt.figure(figsize=(200,5))
plt.scatter(ax,ypred,c='g')
plt.scatter(ax,ytest,c='r')

plt.scatter(ax,t,c='b')
plt.show()