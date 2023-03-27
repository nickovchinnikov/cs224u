# %% [markdown]
# # Homework 1 and bake-off: Word relatedness
# ## Set-up
# %%
from collections import defaultdict
import csv
import itertools
import numpy as np
import os
import pandas as pd
import random
from scipy.stats import spearmanr

import vsm
import utils
from torch_glove import TorchGloVe

# %%
utils.fix_random_seeds()

VSM_HOME = os.path.join('data', 'vsmdata')

DATA_HOME = os.path.join('data', 'wordrelatedness')

# %% [markdown]
# ## Development dataset
# %%
dev_df = pd.read_csv(
    os.path.join(DATA_HOME, "cs224u-wordrelatedness-dev.csv"))

# %%
dev_df.head()

# %%
dev_df.shape[0]

# %% [markdown]
# ### Full dev Vocabulary
# %%
dev_vocab = set(dev_df.word1.values) | set(dev_df.word2.values)
len(dev_vocab)

# %% [markdown]
# The vocabulary for the bake-off test is different
# %%
task_index = pd.read_csv(
    os.path.join(VSM_HOME, 'yelp_window5-scaled.csv.gz'),
    usecols=[0], index_col=0)

full_task_vocab = list(task_index.index)

# %%
len(full_task_vocab)

# %% [markdown]
# ### Score distribution
# %%
ax = dev_df.plot.hist().set_xlabel("Relatedness score")
ax

# %% [markdown]
# ### Repeated pairs
# %%
repeats = dev_df.groupby(['word1', 'word2']).apply(lambda x: x.score.var())

repeats = repeats[repeats > 0].sort_index()

repeats.name = 'score variance'

repeats = repeats.round(3)
repeats.head()

# %%
dev_df.head()

# %% [markdown]
# #### Same solution without apply
# %%
repeats2 = dev_df.groupby(['word1', 'word2']).var()

repeats2 = repeats2[repeats2['score'] > 0].sort_index()

repeats2 = repeats2['score']
repeats2.name = 'score variance'

repeats2 = repeats2.round(3)
repeats2.head()

# %%
len(repeats[repeats2 != repeats])

# %% [markdown]
# ## Evaluation
# %%
count_df = pd.read_csv(
    os.path.join(VSM_HOME, "giga_window5-scaled.csv.gz"), index_col=0)

count_df.head()

# %%
count_pred_df, count_rho = vsm.word_relatedness_evaluation(dev_df, count_df)

# %%
count_pred_df.head()

# %%
count_rho

# %% [markdown]
# ### Error Analysis
# %%
def error_analysis(pred_df):
    pred_df = pred_df.copy()
    pred_df['relatedness_rank'] = _normalized_ranking(pred_df.prediction)
    pred_df['score_rank'] = _normalized_ranking(pred_df.score)
    pred_df['error'] =  abs(pred_df['relatedness_rank'] - pred_df['score_rank']).round(5)
    return pred_df.sort_values('error')


def _normalized_ranking(series):
    ranks = series.rank(method='dense')
    return ranks / ranks.sum()

# %%
error_analysis(count_pred_df)

# %% [markdown]
# ### PPMI as a baseline
# %%
giga_window20 = pd.read_csv(
    os.path.join(VSM_HOME, "giga_window20-flat.csv.gz"), index_col=0)

# %%
giga_window20.head()

# %%
def run_giga_ppmi_baseline():
    GIGA_WINDOW_20 = os.path.join(VSM_HOME, "giga_window20-flat.csv.gz")
    df = pd.read_csv(
        GIGA_WINDOW_20,
        index_col=0
    )
    df_pmi = vsm.pmi(df, positive=True)

    return vsm.word_relatedness_evaluation(dev_df, df_pmi)

# %%
def test_run_giga_ppmi_baseline(func):
    """`func` should be `run_giga_ppmi_baseline"""
    pred_df, rho = func()
    rho = round(rho, 3)
    expected = 0.586
    assert rho == expected, \
        "Expected rho of {}; got {}".format(expected, rho)
    
# %%
if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_giga_ppmi_baseline(run_giga_ppmi_baseline)

# %% [markdown]
# ### Gigaword with LSA
# %%
def run_ppmi_lsa_pipeline(count_df, k):
    count_df_pmi = vsm.pmi(count_df, positive=True)
    count_df_pmi_lsa = vsm.lsa(count_df_pmi, k)
    return vsm.word_relatedness_evaluation(dev_df, count_df_pmi_lsa)

# %%
def test_run_ppmi_lsa_pipeline(func):
    """`func` should be `run_ppmi_lsa_pipeline`"""
    giga20 = pd.read_csv(
        os.path.join(VSM_HOME, "giga_window20-flat.csv.gz"), index_col=0)
    pred_df, rho = func(giga20, k=10)
    rho = round(rho, 3)
    expected = 0.545
    assert rho == expected,\
        "Expected rho of {}; got {}".format(expected, rho)
    
# %%
if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_run_ppmi_lsa_pipeline(run_ppmi_lsa_pipeline)
    
# %% [markdown]
# ### t-test reweighting
# %%
def ttest(df):
    x = df.to_numpy()
    total = x.sum()
    x_prob = x / total

    col_prob = x.sum(axis=0) / total
    row_prob = x.sum(axis=1) / total
    outer_prob = np.outer(row_prob, col_prob)

    return (x_prob - outer_prob) / np.sqrt(outer_prob)


# %%
def test_ttest_implementation(func):
    """`func` should be `ttest`"""
    X = pd.DataFrame([
        [1.,  4.,  3.,  0.],
        [2., 43.,  7., 12.],
        [5.,  6., 19.,  0.],
        [1., 11.,  1.,  4.]])
    actual = np.array([
        [ 0.04655, -0.01337,  0.06346, -0.09507],
        [-0.11835,  0.13406, -0.20846,  0.10609],
        [ 0.16621, -0.23129,  0.38123, -0.18411],
        [-0.0231 ,  0.0563 , -0.14549,  0.10394]])
    predicted = func(X)
    assert np.array_equal(predicted.round(5), actual), \
        "Your ttest result is\n{}".format(predicted.round(5))
    

# %%
if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_ttest_implementation(ttest)


# %% [markdown]
# ### Pooled BERT representation
# %%
from transformers import BertModel, BertTokenizer

def evaluate_pooled_bert(rel_df, layer, pool_func):
    bert_weights_name = 'bert-base-uncased'


    # Initialize a BERT tokenizer and BERT model based on
    # `bert_weights_name`:
    ##### YOUR CODE HERE
    bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    bert_model = BertModel.from_pretrained(bert_weights_name)


    # Get the vocabulary from `rel_df`:
    ##### YOUR CODE HERE
    vocab = set(rel_df['word1'].values) | set(rel_df['word2'].values)


    # Use `vsm.create_subword_pooling_vsm` with the user's arguments:
    ##### YOUR CODE HERE
    pooled_df = vsm.create_subword_pooling_vsm(
        vocab,
        bert_tokenizer,
        bert_model,
        layer=layer,
        pool_func=pool_func
    )


    # Return the results of the relatedness evalution:
    ##### YOUR CODE HERE
    return vsm.word_relatedness_evaluation(
        rel_df,
        pooled_df
    )

# %%
def test_evaluate_pooled_bert(func):
    import torch
    rel_df = pd.DataFrame([
        {'word1': 'porcupine', 'word2': 'capybara', 'score': 0.6},
        {'word1': 'antelope', 'word2': 'springbok', 'score': 0.5},
        {'word1': 'llama', 'word2': 'camel', 'score': 0.4},
        {'word1': 'movie', 'word2': 'play', 'score': 0.3}])
    layer = 2
    pool_func = vsm.max_pooling
    pred_df, rho = func(rel_df, layer, pool_func)
    rho = round(rho, 2)
    expected_rho = 0.40
    assert rho == expected_rho, \
        "Expected rho={}; got rho={}".format(expected_rho, rho)

# %%
if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_evaluate_pooled_bert(evaluate_pooled_bert)

# %% [markdown]
# ### Learned distance function
# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def knn_represent(word1, word2, vsm_df):
    # Use `vsm_df` to get vectors for `word1` and `word2`
    # and concatenate them into a single vector:
    ##### YOUR CODE HERE
    return np.hstack([
        vsm_df.loc[word1].to_numpy(),
        vsm_df.loc[word2].to_numpy()
    ])


def knn_feature_matrix(vsm_df, rel_df):
    # Complete `knn_represent` and use it to create a feature
    # matrix `np.array`:
    ##### YOUR CODE HERE
    return np.vstack([
        knn_represent(w1, w2, vsm_df)
        for w1, w2, _ in rel_df.to_records(index=False)
    ])
    
    

def run_knn_score_model(vsm_df, dev_df, test_size=0.20):
    # Complete `knn_feature_matrix` for this step.
    ##### YOUR CODE HERE
    X = knn_feature_matrix(vsm_df, dev_df)


    # Get the values of the 'score' column in `dev_df`
    # and store them in a list or array `y`.
    ##### YOUR CODE HERE
    y = dev_df['score'].to_numpy()


    # Use `train_test_split` to split (X, y) into train and
    # test protions, with `test_size` as the test size.
    ##### YOUR CODE HERE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y
    )


    # Instantiate a `KNeighborsRegressor` with default arguments:
    ##### YOUR CODE HERE
    model = KNeighborsRegressor()

    # Fit the model on the training data:
    ##### YOUR CODE HERE
    model.fit(X_train, y_train)


    # Return the value of `score` for your model on the test split
    # you created above:
    ##### YOUR CODE HERE
    return model.score(X_test, y_test)

# %%
def test_knn_represent(func):
    vsm_df = pd.DataFrame([
        [1, 2, 3.],
        [4, 5, 6.],
        [7, 8, 9.]], index=['w1', 'w2', 'w3'])
    result = func('w1', 'w3', vsm_df)
    expected = np.array([1, 2, 3, 7, 8, 9.])
    assert np.array_equal(result, expected), \
        "Your `knn_represent` returns: {}\nWe expect: {}".format(
        result, expected)


def test_knn_feature_matrix(func):
    rel_df = pd.DataFrame([
        {'word1': 'w1', 'word2': 'w2', 'score': 0.1},
        {'word1': 'w1', 'word2': 'w3', 'score': 0.2}])
    vsm_df = pd.DataFrame([
        [1, 2, 3.],
        [4, 5, 6.],
        [7, 8, 9.]], index=['w1', 'w2', 'w3'])
    expected = np.array([
        [1, 2, 3, 4, 5, 6.],
        [1, 2, 3, 7, 8, 9.]])
    result = func(vsm_df, rel_df)
    assert np.array_equal(result, expected), \
        "Your `knn_feature_matrix` returns: {}\nWe expect: {}".format(
        result, expected)

  
# %%
if 'IS_GRADESCOPE_ENV' not in os.environ:
    test_knn_represent(knn_represent)
    test_knn_feature_matrix(knn_feature_matrix)


# %% [markdown]
# ### Original system
# %%
def create_count_df():
    return pd.read_csv(
        os.path.join(VSM_HOME, "giga_window5-scaled.csv.gz"),
        index_col=0
    )


# %%
count_df = create_count_df()
count_df.head()

# %%
def glove(count_df):
    glove_model = TorchGloVe()
    return glove_model.fit(count_df)


# %%
count_df_glove = glove(count_df)
count_df_glove.head()


# %%
# GloVe
_, rho = vsm.word_relatedness_evaluation(
    dev_df,
    count_df_glove,
    distfunc=vsm.cosine
)

# 0.5471193870011433
rho


# %%
def lsa(count_df, k=100):
    return vsm.lsa(count_df, k=k)


# %%
count_df_lsa = lsa(count_df, k=100)
count_df_lsa.head()


# %%
def pmi(count_df, positive=True):
    return vsm.pmi(count_df, positive=positive)


# %%
count_df_pmi = pmi(count_df)


# %%
count_df_pmi_lsa = lsa(count_df_pmi)


# %%
count_df_pmi_lsa


# %%
# PMI LSA
_, rho = vsm.word_relatedness_evaluation(
    dev_df,
    count_df_pmi_lsa,
    distfunc=vsm.cosine
)

# 0.6550692841236674
rho


# %%
def t_test(count_df):
    cols = count_df.columns
    indices = count_df.index
    matrix = ttest(count_df)
    return pd.DataFrame(matrix, index=indices, columns=cols)


# %%
count_df_pmi_ttest = t_test(count_df_pmi)


# %%
# PMI TTEST
_, rho = vsm.word_relatedness_evaluation(
    dev_df,
    count_df_pmi_ttest,
    distfunc=vsm.cosine
)

# 0.680917282404449
rho

# %%
