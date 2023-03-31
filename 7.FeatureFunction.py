# %% [markdown]
# ### Setup
# %%
from collections import Counter
import os
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import scipy.stats
import torch.nn as nn

from np_sgd_classifier import BasicSGDClassifier
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

import sst
import utils

# %%
utils.fix_random_seeds()

SST_HOME = os.path.join('data', 'sentiment')

# %% [markdown]
# ### Unigrams
# %%
def unigrams_phi(text):
    """
    The basis for a unigrams feature function. Downcases all tokens.

    Parameters
    ----------
    text : str
        The example to represent.

    Returns
    -------
    defaultdict
        A map from strings to their counts in `text`. (Counter maps a
        list to a dict of counts of the elements in that list.)

    """
    return Counter(text.lower().split())

# %%
example_text = "NLU is enlightening !"
unigrams_phi(example_text)

# %% [markdown]
# ### Bigrams
# %%
def bigrams_phi(text):
    """
    The basis for a bigrams feature function. Downcases all tokens.

    Parameters
    ----------
    text : str
        The example to represent.

    Returns
    -------
    defaultdict
        A map from tuples to their counts in `text`.

    """
    toks = text.lower().split()
    left = [utils.START_SYMBOL] + toks
    right = toks + [utils.END_SYMBOL]
    grams = list(zip(left, right))
    return Counter(grams)

# %%
bigrams_phi(example_text)

# %% [markdown]
# ### DictVectorizer
# %%
train_feats = [
    {'a': 1, 'b': 1},
    {'b': 1, 'c': 2}]

# %%
dic_model = DictVectorizer(sparse=False)
X_train = dic_model.fit_transform(train_feats)

# %%
pd.DataFrame(X_train, columns=dic_model.get_feature_names())

# %% [markdown]
# New data example.
# To transform `test_feats` to `train_feats` space we'll just call `transform`.
# %%
test_feats = [
    {'a': 2, 'c': 1},
    {'a': 4, 'b': 2, 'd': 1}]

# %%
X_test = dic_model.transform(test_feats)

# %%
pd.DataFrame(X_test, columns=dic_model.get_feature_names())

# %% [markdown]
# ### Building datasets for experiments
# %%
train_dataset = sst.build_dataset(
    sst.train_reader(SST_HOME),
    phi=unigrams_phi,
    vectorizer=None
)

# %%
train_dataset['X'].shape

# %% [markdown]
# You can reuse `train_dataset['vectorizer']`
# %%
dev_dataset = sst.build_dataset(
    sst.dev_reader(SST_HOME),
    phi=unigrams_phi,
    vectorizer=train_dataset['vectorizer'])

# %% [markdown]
# ### Wrapper for SGDClassifier
# %%
def fit_basic_sgd_classifier(X, y):
    """
    Wrapper for `BasicSGDClassifier`.

    Parameters
    ----------
    X : np.array, shape `(n_examples, n_features)`
        The matrix of features, one example per row.

    y : list
        The list of labels for rows in `X`.

    Returns
    -------
    BasicSGDClassifier
        A trained `BasicSGDClassifier` instance.

    """
    mod = BasicSGDClassifier()
    mod.fit(X, y)
    return mod

# %%
_ = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_basic_sgd_classifier,
    assess_dataframes=sst.dev_reader(SST_HOME),
    train_size=0.7,
    score_func=utils.safe_macro_f1,
    verbose=True)

# %% [markdown]
# ### Wrapper for LogisticRegression
# %%
def fit_softmax_classifier(X, y):
    """
    Wrapper for `sklearn.linear.model.LogisticRegression`. This is
    also called a Maximum Entropy (MaxEnt) Classifier, which is more
    fitting for the multiclass case.

    Parameters
    ----------
    X : np.array, shape `(n_examples, n_features)`
        The matrix of features, one example per row.

    y : list
        The list of labels for rows in `X`.

    Returns
    -------
    sklearn.linear.model.LogisticRegression
        A trained `LogisticRegression` instance.

    """
    mod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='auto')
    mod.fit(X, y)
    return mod

# %%
_ = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_softmax_classifier)

# %% [markdown]
# ### Wrapper for TorchShallowNeuralClassifier
# %%
def fit_nn_classifier(X, y):
    mod = TorchShallowNeuralClassifier(
        hidden_dim=100,
        early_stopping=True,      # A basic early stopping set-up.
        validation_fraction=0.1,  # If no improvement on the
        tol=1e-5,                 # validation set is seen within
        n_iter_no_change=10)      # `n_iter_no_change`, we stop.
    mod.fit(X, y)
    return mod

# %% [markdown]
# NN experiment
# %%
_ = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_nn_classifier)

# %% [markdown]
# ### A softmax classifier
# %%
class TorchSoftmaxClassifier(TorchShallowNeuralClassifier):
    def build_graph(self):
        return nn.Linear(self.input_dim, self.n_classes_)
    
# %%
def fit_torch_softmax(X, y):
    model = TorchSoftmaxClassifier(l2_strength=1e-4)
    model.fit(X, y)
    return model

# %%
_ = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_torch_softmax)

# %%
# Example of TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Create a list of documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create a CountVectorizer to compute term frequencies
count_vectorizer = CountVectorizer()
term_counts = count_vectorizer.fit_transform(documents)

# %%
term_counts.toarray()

# %%
rescaler = TfidfTransformer()
tfidf_matrix = rescaler.fit_transform(term_counts)
tfidf_matrix.toarray()

# %% [markdown]
# ### Using sklearn Pipelines
# %%
def fit_pipeline_softmax(X, y):
    rescaler = TfidfTransformer()
    mod = LogisticRegression(max_iter=2000)
    pipeline = Pipeline([
        ('scaler', rescaler),
        ('model', mod)])
    pipeline.fit(X, y)
    return pipeline

# %%
_ = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_pipeline_softmax)

# %% [markdown]
# Combine with the models from the course repo
# %%
def fit_pipeline_classifier(X, y):
    rescaler = TfidfTransformer()
    mod = TorchShallowNeuralClassifier(early_stopping=True)
    pipeline = Pipeline([
        ('scaler', rescaler),
        # We need this little bridge to go from
        # sparse matrices to dense ones:
        ('densify', utils.DenseTransformer()),
        ('model', mod)])
    pipeline.fit(X, y)
    return pipeline

# %%
_ = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_pipeline_classifier)

# %% [markdown]
# ## Hyperparameter Search
# Example with Logistic regression
# %%
def fit_softmax_with_hyperparameter_search(X, y):
    """
    A MaxEnt model of dataset with hyperparameter cross-validation.

    Some notes:

    * 'fit_intercept': whether to include the class bias feature.
    * 'C': weight for the regularization term (smaller is more regularized).
    * 'penalty': type of regularization -- roughly, 'l1' ecourages small
      sparse models, and 'l2' encourages the weights to conform to a
      gaussian prior distribution.
    * 'class_weight': 'balanced' adjusts the weights to simulate a
      balanced class distribution, whereas None makes no adjustment.

    Other arguments can be cross-validated; see
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.

    y : list
        The list of labels for rows in `X`.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model instance, the best model found.

    """
    basemod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='auto')
    cv = 5
    param_grid = {
        'C': [0.6, 0.8, 1.0, 2.0],
        'penalty': ['l1', 'l2'],
        'class_weight': ['balanced', None]}
    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, basemod, cv, param_grid)
    return bestmod

# %%
softmax_experiment = sst.experiment(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_softmax_with_hyperparameter_search,
    assess_dataframes=sst.dev_reader(SST_HOME))

# %%
train_df = sst.train_reader(SST_HOME)

train_bin_df = train_df[train_df.label != 'neutral']

# %% [markdown]
# ### Baseline from Socher el al. 2013
# %%
train_df = sst.train_reader(SST_HOME)
    
train_bin_df = train_df[train_df.label != 'neutral']

# %%
dev_df = sst.dev_reader(SST_HOME)

dev_bin_df = dev_df[dev_df.label != 'neutral']

# %%
test_df = sst.sentiment_reader(os.path.join(SST_HOME, "sst3-test-labeled.csv"))

test_bin_df = test_df[test_df.label != 'neutral']

# %%
full_train_df = sst.train_reader(SST_HOME, include_subtrees=True)
full_train_bin_df = full_train_df[full_train_df.label != 'neutral']

split_indices = [0] * full_train_bin_df.shape[0]
split_indices += [-1] * dev_bin_df.shape[0]
sst_train_dev_splitter = PredefinedSplit(split_indices)

# %% [markdown]
# ### Reproducing the Unigram NaiveBayes result 
# %%
def fit_unigram_nb_classifier(X, y):
    mod = MultinomialNB()
    mod.fit(X, y)
    return mod

# %%
_ = sst.experiment(
    train_bin_df,
    unigrams_phi,
    fit_unigram_nb_classifier,
    assess_dataframes=dev_bin_df)

# %% [markdown]
# Fine-tune the hyperparameters
# %%
def fit_nb_classifier_with_hyperparameter_search(X, y):
    rescaler = TfidfTransformer()
    mod = MultinomialNB()

    pipeline = Pipeline([('scaler', rescaler), ('model', mod)])

    # Access the alpha and fit_prior parameters of `mod` with
    # `model__alpha` and `model__fit_prior`, where "model" is the
    # name from the Pipeline. Use 'passthrough' to optionally
    # skip TF-IDF.
    param_grid = {
        'model__fit_prior': [True, False],
        'scaler': ['passthrough', rescaler],
        'model__alpha': [0.1, 0.2, 0.4, 0.8, 1.0, 1.2]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, pipeline,
        param_grid=param_grid,
        cv=5)
    return bestmod

# %%
unigram_nb_experiment_xval = sst.experiment(
    [train_bin_df, dev_bin_df],
    unigrams_phi,
    fit_nb_classifier_with_hyperparameter_search,
    assess_dataframes=test_bin_df)

# %% [markdown]
# ### Reproducing the Bigrams NaiveBayes results
# %%
bigram_nb_experiment_xval = sst.experiment(
    [train_bin_df, dev_bin_df],
    bigrams_phi,
    fit_nb_classifier_with_hyperparameter_search,
    assess_dataframes=test_bin_df)

# %% [markdown]
# ### Reproducing the SVM results
# %%
def fit_svm_classifier_with_hyperparameter_search(X, y):
    rescaler = TfidfTransformer()
    mod = LinearSVC(loss='squared_hinge', penalty='l2')

    pipeline = Pipeline([('scaler', rescaler), ('model', mod)])

    # Access the alpha parameter of `mod` with `mod__alpha`,
    # where "model" is the name from the Pipeline. Use
    # 'passthrough' to optionally skip TF-IDF.
    param_grid = {
        'scaler': ['passthrough', rescaler],
        'model__C': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]}

    bestmod = utils.fit_classifier_with_hyperparameter_search(
        X, y, pipeline,
        param_grid=param_grid,
        cv=5)
    return bestmod

# %%
svm_experiment_xval = sst.experiment(
    [train_bin_df, dev_bin_df],
    unigrams_phi,
    fit_svm_classifier_with_hyperparameter_search,
    assess_dataframes=test_bin_df)

# %% [markdown]
# ### Comparison with the Wilcoxon signed-rank test
# %%
_ = sst.compare_models(
    sst.train_reader(SST_HOME),
    unigrams_phi,
    fit_softmax_classifier,
    stats_test=scipy.stats.wilcoxon,
    trials=10,
    phi2=None,  # Defaults to same as first argument.
    train_func2=fit_basic_sgd_classifier, # Defaults to same as second argument.
    train_size=0.7,
    score_func=utils.safe_macro_f1)

# %%
