# %%
from os import path
import pandas as pd

import vsm
import utils

# %%
DATA_HOME = path.join('data', 'vsmdata')

# %% [markdown]
# ## Load datasets
# %%
# Yelp windows size = 5; scaling = 1/n;
yelp5 = pd.read_csv(
    path.join(DATA_HOME, 'yelp_window5-scaled.csv.gz'), index_col=0
)

# %%
yelp5.head()

# %%
# Yelp windows size = 20; scaling = flat;
yelp20 = pd.read_csv(
    path.join(DATA_HOME, 'yelp_window20-flat.csv.gz'), index_col=0
)

# %%
yelp20.head()

# %%
# Yelp windows size = 5; scaling = 1/n;
giga5 = pd.read_csv(
    path.join(DATA_HOME, 'giga_window5-scaled.csv.gz'), index_col=0
)

# %%
giga5.head()

# %%
# Gigaword windows size = 20; scaling = flat;
giga20 = pd.read_csv(
    path.join(DATA_HOME, 'giga_window20-flat.csv.gz'), index_col=0
)

# %%
giga20.head()

# %% [markdown]
# ## Distributional neighbors
# %%
ABC = pd.DataFrame([
    [ 2.0,  4.0],
    [10.0, 15.0],
    [14.0, 10.0]],
    index=['A', 'B', 'C'],
    columns=['x', 'y'])

ABC
# %% [markdown]
# ### Different distances function for neighbors function
# %%
vsm.neighbors('A', ABC, distfunc=vsm.euclidean)

# %%
vsm.neighbors('A', ABC, distfunc=vsm.cosine)

# %% [markdown]
# ### Check on the yelp5 dataset
# %%
vsm.neighbors('superb', yelp5, distfunc=vsm.euclidean).head()

# %%
vsm.neighbors('superb', yelp20, distfunc=vsm.euclidean).head()

# %% [markdown]
# ### Check cosine distfunc give us much better result!
# %%
vsm.neighbors('superb', yelp5, distfunc=vsm.cosine).head()

# %%
vsm.neighbors('superb', yelp20, distfunc=vsm.cosine).head()

# %%
vsm.neighbors('superb', giga20, distfunc=vsm.cosine).head()

# %% [markdown]
# ### Matrix reweighting
# %%
yelp5_oe = vsm.observed_over_expected(yelp5)
yelp5_oe.head()

# %%
vsm.neighbors('superb', yelp5_oe).head()

# %% [markdown]
# ## Pointwise Mutual Information
# %%
yelp5_pmi = vsm.pmi(yelp5)
yelp5_pmi.head()

# %%
vsm.neighbors('superb', yelp5_pmi).head()

# %% [markdown]
### TF-IDF
# %%
yelp5_tfidf = vsm.tfidf(yelp5)
yelp5_tfidf.head()

# %%
