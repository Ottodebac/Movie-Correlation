#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

df = pd.read_csv('/Users/nicholasr.barton/Desktop/Data/movies.csv')

df.head()


# In[11]:


df.head()


# In[4]:


#check for missing data 

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))
  


# In[5]:


# data types for our columns 

df.dtypes


# In[7]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[11]:


#Makes rows above scrollable 
pd.set_option('display.max_rows',None)


# In[16]:


# Finding coorelation


# In[23]:


# Scatter Plot with Budget vs gross 

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs. Gross Earnings')

plt.xlabel('Gross Earnings')

plt.ylabel('Budget')
                                 
plt.show()


# In[32]:


#plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color": "grey"}, line_kws={"color":"green"})


# In[33]:


# Coorelation 


# In[38]:


df.corr(method='spearman')


# In[7]:


coorelation_matrix = df.corr(method='spearman')

sns.heatmap(coorelation_matrix, annot=True)

plt.show()


# In[ ]:


df_numerized.corr(method='pearson')


# In[ ]:





# In[55]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[ ]:




