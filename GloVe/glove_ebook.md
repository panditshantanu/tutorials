# Understanding GloVe

## A Detailed Tutorial on Global Vectors for Word Representation

------------------------------------------------------------------------

# Table of Contents

1.  Introduction
2.  Why Computers Need Word Representations
3.  Limitations of One‑Hot Encoding
4.  The Distributional Hypothesis
5.  Context Windows
6.  Building the Co‑Occurrence Matrix
7.  Computing Probabilities
8.  The Ratio Idea Behind GloVe
9.  The GloVe Model
10. Training Objective
11. Ambiguous Contexts
12. Implementing GloVe from Scratch in Python
13. Visualizing Embeddings
14. Limitations of GloVe
15. Relationship with Word2Vec
16. Modern Embeddings

------------------------------------------------------------------------

# 1. Introduction

Natural language processing systems must convert words into numerical
form before performing any computation.

Humans naturally understand relationships between words. For example:

    ice
    steam
    water

We intuitively know that these words are related. However, computers
cannot understand these relationships unless we create numerical
representations that capture them.

This is the goal of **word embeddings**.

Word embeddings represent words as vectors of numbers such that:

-   Words with similar meanings have similar vectors
-   Semantic relationships are preserved in vector space

One of the most influential algorithms for learning word embeddings is
**GloVe (Global Vectors for Word Representation)**.

------------------------------------------------------------------------

# 2. Why Computers Need Word Representations

Computers cannot directly process text. Every word must be converted
into numbers.

A naive solution would be assigning numbers:

    ice → 1
    steam → 2
    water → 3
    cold → 4
    hot → 5

But this approach does not capture relationships between words.

For example:

    distance(ice, steam)

has no meaningful interpretation.

Instead we want:

    distance(ice, steam) < distance(ice, airplane)

To achieve this we use **vector representations**.

------------------------------------------------------------------------

# 3. Limitations of One‑Hot Encoding

One of the earliest techniques is **one-hot encoding**.

Suppose our vocabulary is:

    ice
    steam
    water
    cold
    hot

The vectors look like:

  Word    Vector
  ------- ---------------
  ice     \[1,0,0,0,0\]
  steam   \[0,1,0,0,0\]
  water   \[0,0,1,0,0\]
  cold    \[0,0,0,1,0\]
  hot     \[0,0,0,0,1\]

## Problems

### No semantic similarity

    distance(ice, steam) = distance(ice, hot)

Thus the representation contains no meaning.

### High dimensionality

Large vocabularies require extremely large vectors.

------------------------------------------------------------------------

# 4. The Distributional Hypothesis

A central idea in linguistics states:

> Words that appear in similar contexts tend to have similar meanings.

Example:

    The cat sat on the mat
    The dog sat on the mat

Both **cat** and **dog** share the context:

    sat
    on
    the
    mat

From this context we infer they belong to a similar category.

Thus we represent words using **their surrounding contexts**.

------------------------------------------------------------------------

# 5. Context Windows

To capture context we define a **window size**.

Example sentence:

    Ice is cold

Positions:

    1 Ice
    2 is
    3 cold

## Window size = 1

We look one word left and one word right.

Generated pairs:

    (ice, is)
    (is, ice)
    (is, cold)
    (cold, is)

Notice:

    (ice, cold)

does **not appear** because the distance is two.

If window size were **2**, it would appear.

------------------------------------------------------------------------

# 6. Building the Co‑Occurrence Matrix

Consider corpus:

    Ice is cold
    Steam is hot
    Ice is frozen
    Steam is vapor
    Water becomes ice
    Water becomes steam

Example counts:

    X(ice, is) = 2
    X(ice, becomes) = 1
    X(steam, is) = 2
    X(steam, becomes) = 1

Where

    Xij = number of times context word j appears near word i

Example structure:

  Target   Context   Count
  -------- --------- -------
  ice      is        2
  ice      becomes   1
  steam    is        2
  steam    becomes   1

This matrix captures **global statistics of the corpus**.

------------------------------------------------------------------------

# 7. Computing Probabilities

We can compute probabilities from the matrix.

    P(context | word)

Formula:

    P(j|i) = Xij / Xi

Where

    Xi = total context occurrences for word i

Example for **ice**:

    (ice,is) = 2
    (ice,becomes) = 1

Total:

    Xi = 3

Therefore:

    P(is | ice) = 2/3
    P(becomes | ice) = 1/3

------------------------------------------------------------------------

# 8. The Ratio Idea Behind GloVe

The GloVe paper proposes that **ratios of probabilities encode
meaning**.

Example:

    i = ice
    j = steam
    k = cold

Suppose:

    P(cold|ice) = 0.3
    P(cold|steam) = 0.01

Ratio:

    P(cold|ice) / P(cold|steam) = 30

Meaning **cold strongly distinguishes ice from steam**.

Important:

GloVe does **not explicitly compute these ratios** during training.\
They motivate the model mathematically.

------------------------------------------------------------------------

# 9. The GloVe Model

GloVe learns vectors satisfying:

    wi · wj + bi + bj ≈ log(Xij)

Where

-   wi = word vector
-   wj = context vector
-   bi,bj = biases
-   Xij = co‑occurrence count

Meaning:

**vector similarity reflects co‑occurrence frequency**.

------------------------------------------------------------------------

# 10. Training Objective

The loss function is:

    J = Σ f(Xij) (wi·wj + bi + bj − log(Xij))²

Where `f(Xij)` is a weighting function.

Training uses **gradient descent**.

------------------------------------------------------------------------

# 11. Ambiguous Contexts

Words can appear in multiple contexts:

    solid ice
    solid performance
    solid argument

This does not break embeddings because each word has **many contextual
signals**.

Example contexts:

Ice:

    cold
    frozen
    water
    snow
    melt

Performance:

    team
    quarter
    growth
    results

Thus their vectors remain distinct.

------------------------------------------------------------------------

# 12. Implementing GloVe From Scratch in Python

## Step 1 --- Corpus

``` python
sentences = [
"ice is cold",
"steam is hot",
"ice is frozen",
"steam is vapor",
"water becomes ice",
"water becomes steam"
]
```

## Step 2 --- Tokenization

``` python
import re
tokens = [re.findall(r"\w+", s.lower()) for s in sentences]
```

## Step 3 --- Vocabulary

``` python
vocab = sorted(set(word for sentence in tokens for word in sentence))
word_to_id = {w:i for i,w in enumerate(vocab)}
```

## Step 4 --- Co‑Occurrence Matrix

``` python
import numpy as np

window_size = 1
vocab_size = len(vocab)

X = np.zeros((vocab_size,vocab_size))

for sentence in tokens:
    for i,word in enumerate(sentence):

        word_id = word_to_id[word]

        start = max(i-window_size,0)
        end = min(i+window_size+1,len(sentence))

        for j in range(start,end):
            if i!=j:

                context_word = sentence[j]
                context_id = word_to_id[context_word]

                X[word_id,context_id]+=1
```

## Step 5 --- Initialize Embeddings

``` python
embedding_dim = 2

W = np.random.randn(vocab_size,embedding_dim)
W_context = np.random.randn(vocab_size,embedding_dim)

b = np.zeros(vocab_size)
b_context = np.zeros(vocab_size)
```

## Step 6 --- Training Loop

``` python
learning_rate = 0.01
epochs = 500

for epoch in range(epochs):

    for i in range(vocab_size):
        for j in range(vocab_size):

            if X[i,j] > 0:

                xij = X[i,j]

                prediction = np.dot(W[i],W_context[j]) + b[i] + b_context[j]
                target = np.log(xij)

                error = prediction - target

                grad = error

                W[i] -= learning_rate * grad * W_context[j]
                W_context[j] -= learning_rate * grad * W[i]

                b[i] -= learning_rate * grad
                b_context[j] -= learning_rate * grad
```

------------------------------------------------------------------------

# 13. Visualizing Embeddings

``` python
import matplotlib.pyplot as plt

for word in vocab:
    idx = word_to_id[word]

    plt.scatter(W[idx,0],W[idx,1])
    plt.text(W[idx,0],W[idx,1],word)

plt.show()
```

You should observe clusters like:

    ice cold frozen
    steam hot vapor

------------------------------------------------------------------------

# 14. Limitations of GloVe

GloVe produces **one vector per word**.

Example:

    bank

This word has two meanings:

    river bank
    financial bank

The embedding mixes them.

------------------------------------------------------------------------

# 15. Relationship with Word2Vec

  Method     Type
  ---------- ------------------
  Word2Vec   prediction‑based
  GloVe      count‑based

Word2Vec predicts context words.\
GloVe uses global co‑occurrence statistics.

------------------------------------------------------------------------

# 16. Modern Embeddings

Modern models such as:

    BERT
    GPT
    Transformer models

produce **contextual embeddings**, meaning the same word can have
different vectors depending on its context.
