#!/usr/bin/env python
# coding: utf-8

# # Import library

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Import Dataset

# In[2]:


df = pd.read_csv("laptop_data.csv")
df.head()


# In[4]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[12]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[13]:


df.sample()


# In[14]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[15]:


df.sample(10)


# In[16]:


df.info()


# # Data Visualization

# In[19]:


sns.distplot(df['Price'])


# In[20]:


df['Company'].value_counts().plot(kind='bar')


# In[22]:


sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[24]:


sns.countplot(df['TypeName'])
plt.xticks(rotation='vertical')
plt.show()


# In[26]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[27]:


sns.distplot(df['Inches'])


# In[28]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[29]:


df['ScreenResolution'].value_counts()


# In[30]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df.tail()


# In[32]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[36]:


sns.barplot(x=df['Touchscreen'],y=df['Price'], capsize=.2)


# In[37]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df.head()


# In[38]:


sns.countplot(df['Ips'])


# In[39]:


new_cols = df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[40]:


df['X_res'] = new_cols[0]
df['Y_res'] = new_cols[1]


# In[41]:


df.sample(10)


# In[42]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[43]:


df.head()


# In[44]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


# In[45]:


df.info()


# In[46]:


df.corr()['Price']


# In[47]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[49]:


df.corr()['Price']


# In[50]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[51]:


df.sample(5)


# In[52]:


df.drop(columns=['Inches','X_res','Y_res'],inplace=True)


# In[53]:


df.head()


# In[54]:


df['Cpu'].value_counts()


# In[55]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[56]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[57]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[58]:


df.head()


# In[60]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[61]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[62]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[63]:


df.head()


# In[64]:


df['Ram'].value_counts().plot(kind='bar')


# In[65]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[66]:


df['Memory'].value_counts()


# In[69]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)


# In[70]:


df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]=new[1]


# In[74]:


df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)


# In[76]:


df['first'] = df['first'].str.replace(r'\D', '')
df["second"].fillna("0", inplace = True)


# In[79]:


df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)


# In[80]:


df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)


# In[82]:


df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])


# In[83]:


df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid','Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid','Layer2Flash_Storage'],inplace=True)


# In[84]:


df.sample(5)


# In[85]:


df.drop(columns=['Memory'],inplace=True)


# In[86]:


df.head()


# In[ ]:




