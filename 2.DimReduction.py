# %% [markdown]
# ### Setup
# %%
from torch_glove import TorchGloVe
import numpy as np
from np_glove import GloVe
import os
import pandas as pd
import scipy.stats
from torch_autoencoder import TorchAutoencoder
import utils
import vsm

# %%
# Set all the random seeds for reproducibility:

utils.fix_random_seeds()

# %%
DATA_HOME = os.path.join('data', 'vsmdata')

# %%
yelp5 = pd.read_csv(
    os.path.join(DATA_HOME, 'yelp_window5-scaled.csv.gz'), index_col=0)

# %%
yelp20 = pd.read_csv(
    os.path.join(DATA_HOME, 'yelp_window20-flat.csv.gz'), index_col=0)

# %%
giga5 = pd.read_csv(
    os.path.join(DATA_HOME, 'giga_window5-scaled.csv.gz'), index_col=0)

# %%
giga20 = pd.read_csv(
    os.path.join(DATA_HOME, 'giga_window20-flat.csv.gz'), index_col=0)

# %% [markdown]
# ### Latent Semantic Analysis
# %%
gnarly_df = pd.DataFrame(
    np.array([
        [1,0,1,0,0,0],
        [0,1,0,1,0,0],
        [1,1,1,1,0,0],
        [0,0,0,0,1,1],
        [0,0,0,0,0,1]], dtype='float64'),
    index=['gnarly', 'wicked', 'awesome', 'lame', 'terrible'])

gnarly_df

# %%
# Bad result without
vsm.neighbors('gnarly', gnarly_df)

# %%
vsm.neighbors('gnarly', vsm.pmi(gnarly_df))

# %%
gnarly_lsa_df = vsm.lsa(gnarly_df, k=2)

# %%
vsm.neighbors('gnarly', gnarly_lsa_df)

# %% [markdown]
# ### Applying LSA to real VSMs
# %%
vsm.neighbors('superb', yelp5).head()

# %% [markdown]
# And then LSA with $k=100$:
# %%
yelp5_svd = vsm.lsa(yelp5, k=100)

# %%
vsm.neighbors('superb', yelp5_svd).head()

# %%
yelp5_pmi = vsm.pmi(yelp5, positive=False)

# %%
yelp5_pmi_svd = vsm.lsa(yelp5_pmi, k=100)

# %%
vsm.neighbors('superb', yelp5_pmi_svd).head()

# %% [markdown]
# ### GloVe: Global Vectors for Word Representation
# %%
gnarly_glove_mod = GloVe(n=2, max_iter=1000)
gnarly_glove = gnarly_glove_mod.fit(gnarly_df)

# %%
vsm.neighbors('gnarly', gnarly_glove)

# %% [markdown]
# ### Test the GloVo
# %%
glove_test_count_df = pd.DataFrame(
    np.array([
        [10.0,  2.0,  3.0,  4.0],
        [ 2.0, 10.0,  4.0,  1.0],
        [ 3.0,  4.0, 10.0,  2.0],
        [ 4.0,  1.0,  2.0, 10.0]]),
    index=['A', 'B', 'C', 'D'],
    columns=['A', 'B', 'C', 'D'])

# %%
glove_test_mod = GloVe(n=4, max_iter=1000)

# %%
glove_test_df = glove_test_mod.fit(glove_test_count_df)

# %%
glove_test_mod.score(glove_test_count_df)

# %% [markdown]
# ### GloVe and real VSMs
# %%
glove_model = TorchGloVe(max_iter=100)

yelp5_glv = glove_model.fit(yelp5)

# %%
glove_model.score(yelp5)

# %%
vsm.neighbors('superb', yelp5_glv).head()

# %% [markdown]
# ### Autoencoders
# %%
def randmatrix(m, n, sigma=0.1, mu=0):
    return sigma * np.random.randn(m, n) + mu

def autoencoder_evaluation(nrow=1000, ncol=100, rank=20, max_iter=20000):
    """This an evaluation in which `TorchAutoencoder` should be able
    to perfectly reconstruct the input data, because the
    hidden representations have the same dimensionality as
    the rank of the input matrix.
    """
    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    ae = TorchAutoencoder(hidden_dim=rank, max_iter=max_iter)
    ae.fit(X)
    X_pred = ae.predict(X)
    mse = (0.5 * (X_pred - X)**2).mean()
    return(X, X_pred, mse)

# %%
ae_max_iter = 100

_, _, ae = autoencoder_evaluation(max_iter=ae_max_iter)

print("Autoencoder evaluation MSE after {0} evaluations: {1:0.04f}".format(
    ae_max_iter, ae))

# %% [markdown]
# ### Apply autoencodes for real VSMs
# %%
yelp5_l2 = yelp5.apply(vsm.length_norm, axis=1)

# %%
yelp5_l2.head()

# %%
yelp5_l2_ae = TorchAutoencoder(
    max_iter=100, hidden_dim=50, eta=0.001).fit(yelp5_l2)

# %%
yelp5_l2_ae.head()

# %%
vsm.neighbors('superb', yelp5_l2_ae).head()

# %% [markdown]
# To speed-up autoencoder, apply dimensionally reduction technique
# %%
yelp5_l2_svd100 = vsm.lsa(yelp5_l2, k=100)

# %%
yelp_l2_svd100_ae = TorchAutoencoder(
    max_iter=1000, hidden_dim=50, eta=0.01).fit(yelp5_l2_svd100)

# %%
vsm.neighbors('superb', yelp_l2_svd100_ae).head()

# %%
