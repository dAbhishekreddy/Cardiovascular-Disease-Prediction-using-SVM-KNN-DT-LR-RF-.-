#!/usr/bin/env python
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplot', 'inline')


# In[138]:


df=pd.read_csv(r"C:\Users\abhis\Downloads\Project 1 - Cardiovascular Disease Prediction using Machine Learing\cardio_train.csv\cardio_train.csv",sep=';')
df


# In[139]:


df.isnull().values.any()


# In[140]:


sns.scatterplot(x='weight',y='cholesterol',data=df)


# In[141]:


plt.show()


# In[142]:


df_plot=pd.DataFrame({'weight':df['weight'],'cholesterol':df['cholesterol']})


# In[143]:


sns.lineplot(x='weight',y='cholesterol',data=df_plot)


# In[144]:


df_plot=pd.DataFrame({'active':df['active'],'cardio':df['cardio']})


# In[145]:


sns.lineplot(x='active',y='cardio',data=df_plot)
plt.title('Activity vs cardio graph')


# In[146]:


df_bar=df[['gender','cardio']]


# In[147]:


df_bar=df_bar.groupby(['gender','cardio']).size().reset_index(name='count')


# In[148]:


sns.barplot(x='gender',y='count',hue='cardio',data=df_bar)
plt.title('cardiovascular disease by gender')
plt.xlabel('age group')
plt.ylabel('Frequency')
plt.show()


# In[149]:


corr=df.corr()
corr


# # corealtion matrix in heatmap

# In[150]:


corr=df.corr()
plt.figure(figsize=(14,10))
sns.heatmap(corr,annot=True,cmap='coolwarm')


# # LOGISTIC REGRESSION

# In[151]:


from sklearn.model_selection import train_test_split


# In[152]:


x_train,x_test,y_train,y_test=train_test_split(df[["age","gender","cholesterol",]],df["cardio"],train_size=0.9)


# In[153]:


from sklearn.linear_model import LogisticRegression
lrmodel=LogisticRegression()


# In[154]:


lrmodel.fit(x_train,y_train)


# In[155]:


lrmodel.predict(x_test)


# In[156]:


len(x_train)


# In[157]:


len(x_test)


# In[158]:


lrmodel.score(x_test,y_test)


# In[159]:


lrmodel.predict([[18393,2,1]])


# # support vector machine 

# In[160]:


df_selected=df[["age","gender","cholesterol","cardio"]]
df_selected


# In[161]:


x=df.iloc[:,1:3]
x


# In[162]:


y=df[["cardio"]]
y


# In[163]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)


# In[164]:


from sklearn.svm import SVC
svm_model=SVC(kernel="linear")
svm_model.fit(x_train,y_train)


# In[165]:


y_pred=svm_model.predict(x_test)
y_pred


# In[166]:


svm_model.score(x_test,y_test)


# # KNN CLASSIFICATION

# In[167]:


x2=df[["weight","smoke","alco"]]
x2


# In[168]:


y2=df[["cardio"]]
y2


# In[169]:


from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,train_size=0.8)


# In[170]:


from sklearn.neighbors import KNeighborsClassifier
knnmodel=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knnmodel.fit(x2_train,y2_train)


# In[171]:


knnmodel.score(x2_test,y2_test)


# In[172]:


knnmodel.predict(x2_test)


# # NAIVE BAYES

# In[173]:


df["gender"].replace({1:"Female",2:"Male"},inplace=True)
df


# In[174]:


x3=df[["weight","smoke","alco"]]
x3


# In[175]:


y3=df[["cardio"]]
y3


# In[176]:


from sklearn.model_selection import train_test_split
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,train_size=0.8)


# In[177]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB
m=GaussianNB()
m1=MultinomialNB()
m.fit(x3_train,y3_train)
m1.fit(x3_train,y3_train)


# In[178]:


m.predict(x3_test)


# In[179]:


m.score(x3_test,y3_test)

m1.score(x3_test,y3_test)
# # DECISION TREE 

# In[181]:


x4=df[["ap_hi","ap_lo","gluc"]]
x4


# In[182]:


y4=df[["cardio"]]
y4


# In[183]:


from sklearn.model_selection import train_test_split
x4_train,x4_test,y4_train,y4_test=train_test_split(x4,y4,train_size=0.8)


# In[184]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[185]:


model.fit(x4_train,y4_train)


# In[186]:


model.score(x4_test,y4_test)


# In[187]:


model.predict(x4_test)


# In[188]:


from sklearn import tree
tree.plot_tree(model)


# # RANDOM FOREST

# In[189]:


x5=df[["ap_hi","ap_lo","gluc"]]
x5


# In[190]:


y5=df[["cardio"]]
y5


# In[191]:


from sklearn.model_selection import train_test_split
x5_train,x5_test,y5_train,y5_test=train_test_split(x5,y5,train_size=0.8)


# In[192]:


from sklearn.ensemble import RandomForestClassifier
rm=RandomForestClassifier()
rm.fit(x5_train,y5_train)


# In[193]:


rm.score(x5_test,y5_test)


# # WE GET THE HIGHEST ACCURACY FROM THE RANDOM FOREST ALGORITHM SO WE CAN USE IT TO BUILD THE MODEL
