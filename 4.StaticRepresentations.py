# %% [markdown]
# ### Setup
# %%
import os
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer

import utils
import vsm

# %%
DATA_HOME = os.path.join('data', 'vsmdata')

utils.fix_random_seeds()

# %%
import logging
logger = logging.getLogger()
logger.level = logging.ERROR

# %%
bert_weights_name = 'bert-base-cased'

# %% [markdown]
# ### Loading transformer model
# %%
bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)

# %%
bert_model = BertModel.from_pretrained(bert_weights_name)

# %% [markdown]
# ### The basic of Tokenizing
# %%
example_text = "Bert knows Snuffleupagus"

# %%
bert_tokenizer.tokenize(example_text)

# %%
ex_ids = bert_tokenizer.encode(example_text, add_special_tokens=True)

ex_ids

# %%
# Convert ids back to tokens
bert_tokenizer.convert_ids_to_tokens(ex_ids)

# %% [markdown]
# ### The basic of representations
# %%
with torch.no_grad():
    reps = bert_model(torch.tensor([ex_ids]), output_hidden_states=True)

# %%
reps.last_hidden_state.shape

# %%
len(reps.hidden_states)

# %%
torch.equal(reps.hidden_states[-1], reps.last_hidden_state)

# %% [markdown]
# ### The decontextualized approach
# %%
bert_tokenizer.tokenize('puppy')

# %%
vsm.hf_encode("puppy", bert_tokenizer)

# %% [markdown]
# Some words maps to multiple tokens
# %%
bert_tokenizer.tokenize('snuffleupagus')

# %%
subtok_ids = vsm.hf_encode("snuffleupagus", bert_tokenizer)

subtok_ids

# %%
subtok_reps = vsm.hf_represent(subtok_ids, bert_model, layer=-1)

subtok_reps.shape

# %%
subtok_pooled = vsm.mean_pooling(subtok_reps)

subtok_pooled.shape

# %% [markdown]
# ### Creating a full VSM
# %%
vsm_index = pd.read_csv(
    os.path.join(DATA_HOME, 'yelp_window5-scaled.csv.gz'),
    usecols=[0], index_col=0)

# %%
vocab = list(vsm_index.index)

# %%
vocab[:5]

# %%
pooled_df = vsm.create_subword_pooling_vsm(
    vocab, bert_tokenizer, bert_model, layer=1)

# %%
pooled_df.shape

# %%
pooled_df.iloc[: 5, :5]

# %% [markdown]
# ### The aggregated approach
# %%
vocab_ids = {w: vsm.hf_encode(w, bert_tokenizer)[0] for w in vocab}

# %%
corpus = [
    "This is a sailing example",
    "It's fun to go sailing!",
    "We should go sailing.",
    "I'd like to go sailing and sailing",
    "This is merely an example"]

# %%
corpus_ids = [vsm.hf_encode(text, bert_tokenizer)
              for text in corpus]

# %%
corpus_reps = [vsm.hf_represent(ids, bert_model, layer=1)
               for ids in corpus_ids]

# %%
def find_sublist_indices(sublist, mainlist):
    indices = []
    length = len(sublist)
    for i in range(0, len(mainlist)-length+1):
        if mainlist[i: i+length] == sublist:
            indices.append((i, i+length))
    return indices

# %%
find_sublist_indices([1,2], [1, 2, 3, 0, 1, 2, 3])

# %%
sailing = vocab_ids['sailing']

# %%
sailing_reps = []

for ids, reps in zip(corpus_ids, corpus_reps):
    offsets = find_sublist_indices(sailing, ids.squeeze(0))
    for (start, end) in offsets:
        pooled = vsm.mean_pooling(reps[:, start: end])
        sailing_reps.append(pooled)

sailing_rep = torch.mean(torch.cat(sailing_reps), axis=0).squeeze(0)

# %%
sailing_rep.shape

# %%
