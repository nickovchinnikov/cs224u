# %% [markdown]
# ### Setup
# %%
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
import os
import pandas as pd

import sst

# %%
SST_HOME = os.path.join('data', 'sentiment')

# %% [markdown]
# ### Train split
# %%
train_df = sst.train_reader(SST_HOME)

# %%
train_df.sample(3, random_state=1)

# %%
train_df.shape[0]

# %% [markdown]
# ### Label distribution
# %%
train_df.label.value_counts()

# %% [markdown]
# Remove the duplicated items
# %%
dup_train_df = sst.train_reader(SST_HOME, dedup=True)

# %%
dup_train_df.shape[0]

# %% [markdown]
# Distribution of examples by length in chars
# %%
_ = train_df.sentence.str.len().hist().set_ylabel("Length in characters")

# %% [markdown]
# Distribution by word count
# %%
train_df['word_count'] = train_df.sentence.str.split().apply(len)
_ = train_df['word_count'].hist().set_ylabel("Length in words")

# %%
_ = train_df.boxplot("word_count", by="label")

# %% [markdown]
# ### Including subtrees
# %%
subtree_train_df = sst.train_reader(SST_HOME, include_subtrees=True)

# %%
subtree_train_df.shape[0]

# %%
subtree_train_df.head()

# %%
subtree_train_df["word_count"] = subtree_train_df.sentence.str.split().apply(len)

_ = subtree_train_df["word_count"].hist().set_ylabel("Length in words")

# %% [markdown]
# Removing a duplicates has a large effect in this settings
# %%
subtree_train_df_dedup = sst.train_reader(
    SST_HOME, include_subtrees=True, dedup=True
)

# %%
subtree_train_df_dedup.shape

# %% [markdown]
# Label distribution
# %%
subtree_train_df_dedup.label.value_counts()

# %% [markdown]
# ### Dev and Test split
# %%
dev_df = sst.dev_reader(SST_HOME)

# %%
dev_df.shape

# %%
dev_df.label.value_counts()

# %%
_ = dev_df.sentence.str.len().hist().set_ylabel("Length in characters")

# %%
dev_df['word_count'] = dev_df.sentence.str.split().apply(len)

_ = dev_df['word_count'].hist().set_ylabel("Length in words")

# %%
_ = dev_df.boxplot("word_count", by="label")

# %% [markdown]
# ### Tokenization
# %%
ex = train_df.iloc[0].sentence

ex

# %%
detokenizer = TreebankWordDetokenizer()

# %%
def detokenize(s):
    return detokenizer.detokenize(s.split())

# %%
detokenize(ex)


# %% [markdown]
# ### Tokenize data
# %%
tokenizer = TreebankWordTokenizer()

# %%
def treebank_tokenize(s):
    return tokenizer.tokenize(s)

# %%
treebank_tokenize("The Rock isn't the new ``Conan'' – he's this generation's Olivier!")

# %%
