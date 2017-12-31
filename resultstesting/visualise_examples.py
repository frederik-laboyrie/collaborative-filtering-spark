
# coding: utf-8

# In[67]:


import pickle
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

DATA_PATH = '/home/frederik/gitdisst/hand-orientation-inference/modelling/'


# In[68]:


with open('good_vanilla_indices.pickle', 'rb') as handle:
    vanilla_good = pickle.load(handle)
    
with open('bad_vanilla_indices.pickle', 'rb') as handle:
    vanilla_bad = pickle.load(handle)


# In[69]:


full_res = np.load(DATA_PATH + 'AllBW.npy')
med_res = np.load(DATA_PATH + 'AllImagesBW64.npy')
low_res = np.load(DATA_PATH + 'AllImagesBW32.npy')


# In[70]:


test_index = int(len(full_res) * 0.8)
full = np.array([np.rot90(x, k=3) for x in full_res[test_index:]])
med = np.array([np.rot90(x, k=3) for x in med_res[test_index:]])
low = np.array([np.rot90(x, k=3) for x in low_res[test_index:]])


# In[71]:


def extract_index_sample(indices_dict, sample_size=5):
    gd_e = indices_dict['vanilla'][0]['good_elev'][0][:sample_size]
    gd_z = indices_dict['vanilla'][1]['good_zen'][0][:sample_size]
    bd_e = indices_dict['vanilla'][2]['bad_elev'][0][:sample_size]
    bd_z = indices_dict['vanilla'][3]['bad_zen'][0][:sample_size]
    return gd_e, gd_z, bd_e, bd_z


# In[72]:


gd_e, gd_z, bd_e, bd_z = extract_index_sample(vanilla_good)


# In[73]:


plt.imshow(full[gd_e[0]], cmap='gray')


# In[74]:


plt.imshow(full[bd_e[0]], cmap='gray')


# In[75]:


plt.imshow(full[gd_z[0]], cmap='gray')


# In[79]:


plt.imshow(full[bd_z[0]], cmap='gray')

