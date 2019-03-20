


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')




df=pd.read_csv('movies.csv', encoding='latin1')



df.head()




def isyear(x):
    if(x==2016):
        return 1
    else :
        return 0
df["is2016"]=df.year.apply(isyear)





#cleaning the data 
df.budget[df.budget==0] =np.mean(df['budget'])
df.budget.describe()



#features and clustering
df['time'] = pd.to_datetime(df['released'])
df['date'],df['month'],df['DayOfweek']=df['time'].dt.day,df['time'].dt.month,df['time'].dt.dayofweek

l=df.groupby('company')['budget'].mean()
df=df.replace({"company": l})
print(df.company.describe())
m=df.groupby('director')['budget'].mean()
df=df.replace({"director": m})
print(df.director.describe())
n=df.groupby('star')['budget'].mean()
df=df.replace({"star": n})
print(df.star.describe())



o=df.groupby('writer')['budget'].mean()
df=df.replace({"writer": o})
print(df.writer.describe())

p=df.groupby('genre')['budget'].mean()
df=df.replace({"genre": p})
print(df.genre.describe())
 




km = KMeans(n_clusters=40)
km.fit(df.loc[((df['company']<=2.450000e+08) & (df['director']<= 2.600000e+08)& (df['star']<=2.500000e+08))][['company','director','star']])
df['cluster1'] = km.predict(df[['company','director','star']])
km.fit(df.loc[((df['director']<= 2.600000e+08)& (df['star']<= 2.500000e+08))][['director','star']])
df['cluster2'] = km.predict(df[['director','star']])
km.fit(df.loc[((df['director']<= 2.600000e+08)&(df['writer']<=  2.287500e+08))][['director','writer']])
df['cluster3'] = km.predict(df[['director','writer']])
km.fit(df.loc[((df['star']<=2.500000e+08) & (df['genre']<=  6.858626e+07))][['star','genre']])
df['cluster4'] = km.predict(df[['star','genre']])
km.fit(df.loc[((df['budget']<=3.000000e+08)& (df['company']<=2.450000e+08))][['budget','company']])
df['cluster5'] = km.predict(df[['budget','company']])
km.fit(df.loc[((df['director']<= 2.600000e+08)&(df['genre']<= 6.858626e+07))][['director','writer']])
df['cluster6'] = km.predict(df[['director','genre']])
df['cc']=(df['runtime']/10)-4



#visulation of genre and rating
count=0 
l=df.groupby(['genre'])['gross'].apply(list).to_dict()
#l=df.groupby(['rating'])['gross'].apply(list).to_dict()
for key in l: 
    for i in l[key]: 
        count=count+i 
    count=count/len(l[key]) 
    l[key]=count 
    count=0

names = list(l.keys()) 
values = list(l.values())




plt.bar(range(len(l)), list(l.values()), align='center')
plt.xticks(range(len(l)),names, fontsize=5, rotation=30)
plt.savefig('bar10.png')
plt.show()


#visualization of clusters , year,score,votes 
c1=pd.unique(df['cluster1'])
c2=pd.unique(df['cluster2'])
c3=pd.unique(df['cluster4'])
c4=pd.unique(df['cluster5'])
c5=pd.unique(df['cluster6'])
c6=sorted(pd.unique(df['year']))
c7=sorted(pd.unique(df['score']))
c8=sorted(pd.unique(df['votes']))

avg_gross1 = [np.mean(df[df['cluster1']==combination]['gross'])/1000000 for combination in c1]
avg_gross2 = [np.mean(df[df['cluster2']==combination]['gross'])/1000000 for combination in c2]
avg_gross3 =[np.mean(df[df['cluster4']==combination]['gross'])/1000000 for combination in c3]
avg_gross4 =[np.mean(df[df['cluster5']==combination]['gross'])/1000000 for combination in c4]
avg_gross5 =[np.mean(df[df['cluster6']==combination]['gross'])/1000000 for combination in c5]
avg_gross6 = [np.mean(df[df['year']==combination]['gross'])/1000000 for combination in c6]
avg_gross7 = [np.mean(df[df['score']==combination]['gross'])/1000000 for combination in c7]
avg_gross8 = [np.mean(df[df['votes']==combination]['gross'])/1000000 for combination in c8]

# plot and stuff
plt.subplot(3,3,1)
plt.ylabel('Revenue (in millions)')
plt.xlabel('Company-Director-Star')
plt.bar(c1, avg_gross1,color='r')

#plt.subplot(3,3,2)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('Director-Star')
#plt.bar(c2, avg_gross2,color='r')

#plt.subplot(3,3,3)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('Star-genre')
#plt.bar(c3, avg_gross3,color='r')

#plt.subplot(3,3,4)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('budget-Company')
#plt.bar(c4, avg_gross4,color='r')

#plt.subplot(3,3,5)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('director-genre')
#plt.bar(c5, avg_gross5,color='r')

#plt.subplot(3,3,6)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('year')
#plt.plot(c6, avg_gross6,color='r')

#plt.subplot(3,3,7)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('year')
#plt.plot(c7, avg_gross7,color='r')

#plt.subplot(3,3,8)
#plt.ylabel('Revenue (in millions)')
#plt.xlabel('year')
#plt.plot(c8, avg_gross8,color='r')






#model Building
dummylist=['cluster1','cluster2','cluster4','cluster5','cluster6','country','genre','rating']
def get_d(df,l):
    for x in l:
        dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
        df=df.drop(x,1)
        df=pd.concat([df,dummies],axis=1)
    return df
df=get_d(df,dummylist)





#predicting on movies of year 2016
train=df[df["is2016"]!=1]
test=df[df["is2016"]==1]



#training and testing

x_train=train.drop(['time','gross','name','released','company','director','star','writer','runtime','is2016','cluster3'], axis=1)
y_train=train['gross']

x_test=test.drop(['time','gross','name','released','company','director','star','writer','runtime','is2016','cluster3'],axis=1)


y_test=test['gross']
y_test


# In[30]:


#RandomForestRegressor model
regr = RandomForestRegressor(max_depth=70, random_state=0,n_estimators=500)
regr.fit(x_train, y_train)
#regr.score(x_test,y_test)
y_pred=regr.predict(x_test)
y_pred=np.log(y_pred)
y_test=np.log(y_test)
mean_squared_error(y_test, y_pred)




#GradientBoostingRegressor model
params = {'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 2,'learning_rate': 0.01}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(x_train, y_train)
clf.score(x_test,y_test)



with open ('sub','wb') as f:
    pickle.dump(regr,f)

