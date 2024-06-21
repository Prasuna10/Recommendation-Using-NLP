#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


movies= pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# In[6]:


credits.head()


# In[7]:


movies.head()


# In[8]:


credits.shape


# In[9]:


movies.shape


# In[10]:


credits1 = credits.rename(index=str, columns={"movie_id":"id"})
df= movies.merge(credits1, on="id")
df.head()


# In[11]:


clean_df= df.drop(columns=['homepage','title_x','title_y','status','production_countries'])
clean_df.head()


# In[12]:


clean_df.info()


# In[13]:


clean_df.head(1)['overview']


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                   token_pattern=r'\w{1,}',ngram_range=(1,3),stop_words='english')
clean_df['overview']=clean_df['overview'].fillna('')


# In[15]:


tf_matrix= tf.fit_transform(clean_df['overview'])
tf_matrix


# In[16]:


tf_matrix.shape


# In[17]:


#sigmoid is used to transform your data b/w 0 to 1
from sklearn.metrics.pairwise import sigmoid_kernel

sig=sigmoid_kernel(tf_matrix,tf_matrix)
sig[0]


# In[18]:


#reverse mapping the indices and movie title
indices= pd.Series(clean_df.index, index=clean_df['original_title']).drop_duplicates()
indices


# In[19]:


indices['Spectre']


# In[20]:


sig[2]


# In[21]:


list(enumerate(sig[indices['Spectre']]))


# In[22]:


sorted(list(enumerate(sig[indices['Spectre']])), key= lambda x:x[1], reverse=True)


# In[23]:


def recommend(title, sig=sig):
    #index of original title
    index= indices[title]
    
    #get the pairwise similarity
    sig_scores= list(enumerate(sig[index]))
    
    #sort them
    sig_scores= sorted(sig_scores, key= lambda x:x[1], reverse=True)
    
    #get the top 10 scores of movies
    sig_scores=sig_scores[1:11]
    
    #get the movie indices
    movie_indices= [i[0] for i in sig_scores]
    
    #titles of top 10 movies
    return clean_df['original_title'].iloc[movie_indices]


# In[24]:


recommend("The Book of Life")

