# %% [markdown]
# ### Set-up
# %%
from collections import defaultdict
from nltk.corpus import wordnet as wn
import numpy as np
import os
import pandas as pd
import retrofitting
from retrofitting import Retrofitter
import utils

# %%
data_home = 'data'

# %%
import nltk

nltk.download("wordnet")
nltk.download('omw-1.4')

# %% [markdown]
# ### Examples
# %%
Q_hat = pd.DataFrame(
    [[0.0, 0.0],
     [0.0, 0.5],
     [0.5, 0.0]],
    columns=['x', 'y'])

Q_hat
# %% [markdown]
# ### Only 0 node has edges
# %%
edges_0 = {0: {1, 2}, 1: set(), 2: set()}

_ = retrofitting.plot_retro_path(Q_hat, edges_0)

# %% [markdown]
# ### All nodes connected to the others
# %%
edges_all = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}

_ = retrofitting.plot_retro_path(Q_hat, edges_all)

# %% [markdown]
# ### All nodes connected to all others, but $\alpha = 0$
# %%
_ = retrofitting.plot_retro_path(
    Q_hat, edges_all,
    retrofitter=Retrofitter(alpha=lambda x: 0))

# %% [markdown]
# ### WordNet and VSM
# %%
def get_wordnet_edges():
    edges = defaultdict(set)
    for ss in wn.all_synsets():
        lem_names = {lem.name() for lem in ss.lemmas()}
        for lem in lem_names:
            edges[lem] |= lem_names
    return edges

# %%
wn_edges = get_wordnet_edges()

# %%
wn_edges

# %% [markdown]
### Reproducing the WordNet synonym graph experiment
# %%
glove_dict = utils.glove2dict(
    os.path.join(data_home, 'glove.6B', 'glove.6B.300d.txt'))

# %%
X_glove = pd.DataFrame(glove_dict).T
X_glove.T.shape

# %%
def convert_edges_to_indices(edges, Q):
    lookup = dict(zip(Q.index, range(Q.shape[0])))
    index_edges = defaultdict(set)
    for start, finish_nodes in edges.items():
        s = lookup.get(start)
        if s:
            f = {lookup[n] for n in finish_nodes if n in lookup}
            if f:
                index_edges[s] = f
    return index_edges

# %%
wn_index_edges = convert_edges_to_indices(wn_edges, X_glove)

# %%
wn_retro = Retrofitter(verbose=True)

# %%
X_retro = wn_retro.fit(X_glove, wn_index_edges)

# %%
