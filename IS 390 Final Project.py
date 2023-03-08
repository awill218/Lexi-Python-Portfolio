#!/usr/bin/env python
# coding: utf-8

# Alexis Williams
# IS 390- Class Project
# 

# # **Stroke Dataset**

# 1. id: unique identifier
# 2. gender: "Male", "Female" or "Other"
# 3. age: age of the patient
# 4. hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 5. heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# 6. ever_married: "No" or "Yes"
# 7. work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 8. Residence_type: "Rural" or "Urban"
# 9. avg_glucose_level: average glucose level in blood
# 10. bmi: body mass index
# 11. smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
# 12. stroke: 1 if the patient had a stroke or 0 if not

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[29]:


df = pd.read_csv('stroke.csv')
df.head()


# In[30]:


df.shape


# In[31]:


df.isna().sum()


# In[32]:


df = df.dropna()

df.isna().sum()


# In[33]:


df['gender'].unique()


# In[34]:


df = df.drop(df[df['gender'] == "Other"].index)
df['gender'].unique()


# In[35]:


df.shape


# In[36]:


print(df['gender'].value_counts()['Male'])
print(df['gender'].value_counts()['Female'])


# ### Question- <u>How likely is a female of any age compared to a male of any age in an urban residence type to have a stroke with a glucose of 100+</u>

# In[ ]:





# In[37]:


df.stroke.value_counts().plot(kind='bar', color=['darkorchid', 'darkseagreen'])
plt.legend(loc = 'upper right', labels =["No stroke", "Stroke"])


# In[38]:


df['stroke'].value_counts()


# Interpretation : of the 4,908 samples, less than 500 people have had a stroke

# In[39]:


df.describe()[['age', 'avg_glucose_level']]


# In[40]:


sns.distplot(df[df['stroke']== 0]['avg_glucose_level'], color = 'darkslategrey')
sns.distplot(df[df['stroke']== 1]['avg_glucose_level'], color = 'violet')
plt.legend(loc = 'upper right', labels =["No stroke", "Stroke"])


# Interpretation: a normal glucose level for a non diabetic is about 90-140Mg/DL. Those with an elevated glucose level are more likely to have a stroke. Majority of people within the normal range did not have a stroke.

# In[41]:


sns.catplot(x= "gender", y="stroke", hue ="Residence_type",
            kind= "bar", data = df)
plt.show()


# There isn't much of a gap between Urban and Rural residence types, but the frequency of strokes is higher in urban areas most likely because of pollution. It also seems that the frequency of strokes is slightly higher for males which is probably because of higher rates of induced stress in males than in females

# In[42]:


sns.distplot(df[df['stroke']==0]['age'], color = "salmon")
sns.distplot(df[df['stroke']== 1]['age'], color = "mediumaquamarine")
plt.legend(loc = 'upper left', labels =["No stroke", "Stroke"])


# Interpretation: age is a contributing feature to having a stroke. As people get older, the risk of stroke increases.

# ## Data Preprocessing

# 

# In[43]:


df1 = df[[ 'gender', 'age', 'Residence_type', 'avg_glucose_level', 'stroke']]
df1


# In[ ]:





# In[44]:


df1.describe()[['age', 'avg_glucose_level']].T[['min', 'max']]


# normalize the numerical values to be between 0 and 1 and Convert the remaing categorical to binary

# In[45]:


num = ['avg_glucose_level', 'age']
scaler = StandardScaler()
df1[num]= scaler.fit_transform(df1[num])
df1.head()


# In[46]:


df1['gender'] = np.where(df1['gender']=='Female', 1, 0)
df1['Residence_type'] = np.where(df['Residence_type'] == 'Urban', 1, 0)
df1


# In[47]:


sns.heatmap(df1.corr(), cmap ="Blues_r", data = df1)


# In[48]:


df1.corr()


# In[49]:


column = ['gender', 'age', 'Residence_type', 'avg_glucose_level']
for column in column:
    lab_enc = preprocessing.LabelEncoder()
    df1[column] = lab_enc.fit_transform(df1[column])


# In[50]:


for column in ['age', 'avg_glucose_level']:
    df1[column] = df1[column]/df1[column].max()


# In[51]:


values = df1.drop(columns = ['stroke'])

label = df1['stroke']

train_df = df1[:2000]
test_df = df1[2908:]


# In[52]:


features =['gender', 'age', 'Residence_type', 'avg_glucose_level']
#split data into train and test set
train_X = np.array(train_df[['gender', 'age', 'Residence_type', 'avg_glucose_level']])
train_y = np.array(train_df['stroke'])
test_X = np.array(train_df[['gender', 'age', 'Residence_type', 'avg_glucose_level']])
test_y = np.array(test_df['stroke'])


# In[53]:


model_comparison = {}


# **Random Forest Classifier**

# In[54]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

rf = RandomForestClassifier()
rf.fit(train_X, train_y)
prediction = rf.predict(test_X)
model_comparison['RandomForestClassifier'] = [metrics.accuracy_score(test_y, prediction),
                                              f1_score(test_y, prediction,average='weighted')]

print(classification_report(test_y, prediction, zero_division = 0))


# **KNN Classification**

# In[55]:


knn_mod = neighbors.KNeighborsClassifier(5)
knn_mod.fit(train_X, train_y)
predictions = knn_mod.predict(test_X)
model_comparison['KNeighborsClassifier'] = [metrics.accuracy_score(test_y, prediction), f1_score(test_y, prediction,average='weighted')]

print(classification_report(test_y, prediction, zero_division=0))


# In[56]:


#from sklearn.linear_model import LogisticRegression

#model = LogisticRegression(max_iter = 1000)
#model.fit(train_X, train_y)
#prediction = model.predict(test_X)
#model_comparison['LogisticRegression'] = [accuracy_score(test_y, prediction), f1_score(test_y, prediction,average='weighted')]

#print(classification_report(prediction, test_y))


# In[57]:


#from sklearn import tree
#from sklearn import model_selection

#tree_mod= tree.DecisionTreeClassifier(max_depth = 5)
#scorer = metrics.make_scorer(metrics.cohen_kappa_score)
#grid = {'max_depth': [1,2,3,4,5,6]}

#gridsearch = model_selection.GridSearchCV(tree_mod, grid).fit(train_X,train_y)
#prediction = gridsearch.predict(test_X)
#model_comparison['DecisionTreeClassifier']= [accuracy_score(test_y,predictions), f1_score(test_y,prediction, average='weighted')]

#print(classification_report(test_y, prediction))


# In[58]:


model_comparison_df = pd.DataFrame.from_dict(model_comparison).T

model_comparison_df.columns = ['Accuracy', 'F1 Score']
model_comparison_df = model_comparison_df.sort_values('F1 Score', ascending=True)
model_comparison_df.style.background_gradient(cmap='Blues')


# In[ ]:




