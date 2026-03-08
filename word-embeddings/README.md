# Word Embeddings – Turning Words into Numbers

## What This Tutorial Covers

1. [Why do we need word embeddings?](#1-why-do-we-need-word-embeddings)
2. [One-Hot Encoding and its limitations](#2-one-hot-encoding-and-its-limitations)
3. [The core idea: words as points in space](#3-the-core-idea-words-as-points-in-space)
4. [Word2Vec – Learning word vectors from text](#4-word2vec)
5. [GloVe – Global Vectors for Word Representation](#5-glove)
6. [Hands-on code examples](#6-hands-on-code-examples)

---

## 1. Why Do We Need Word Embeddings?

Computers cannot process raw text. They only understand numbers. So the very first question when doing any NLP task is:

> **How do we represent a word as a number (or a set of numbers)?**

The simplest idea is to assign every word a unique integer:

```
apple  → 0
banana → 1
cat    → 2
dog    → 3
```

This works for storage, but it creates a problem: the numbers imply a ranking. Is `dog` (3) really "greater than" `apple` (0) in any meaningful way? No. These integers have no semantic meaning.

We need a representation that captures **meaning and relationships** between words.

---

## 2. One-Hot Encoding and Its Limitations

One-hot encoding solves the ranking problem. Each word is a vector of zeros with a single `1` at the word's position:

```
Vocabulary: [apple, banana, cat, dog]

apple  → [1, 0, 0, 0]
banana → [0, 1, 0, 0]
cat    → [0, 0, 1, 0]
dog    → [0, 0, 0, 1]
```

### Problem 1 – High Dimensionality

If the vocabulary has 100,000 words, each vector has 100,000 dimensions. That is enormous and inefficient.

### Problem 2 – No Semantic Similarity

Look at `cat` and `dog`:

```
cat → [0, 0, 1, 0]
dog → [0, 0, 0, 1]
```

The distance between them is the same as the distance between `apple` and `dog`. But we **know** that `cat` and `dog` are semantically much closer (both are animals) than `apple` and `dog`.

One-hot encoding treats every word as equally distant from every other word. It captures **no meaning**.

---

## 3. The Core Idea: Words as Points in Space

What if we represented every word as a **dense, low-dimensional vector** (say, 100 or 300 numbers)?

```
apple  → [0.50,  0.12, -0.73, ...]   # 300 numbers
banana → [0.48,  0.15, -0.70, ...]   # 300 numbers
cat    → [-0.20, 0.80,  0.30, ...]   # 300 numbers
dog    → [-0.22, 0.78,  0.33, ...]   # 300 numbers
```

Now `cat` and `dog` have similar numbers → they are **close in vector space**.
`apple` and `dog` have very different numbers → they are **far apart**.

**This is a word embedding**: each word is a point in a high-dimensional space, and words with similar meanings are near each other.

The famous analogy that falls out of this:

```
king - man + woman ≈ queen
```

This works because the **direction** "man → woman" in the space represents gender, and that same direction can be applied to "king" to reach "queen".

---

## 4. Word2Vec

Word2Vec (Mikolov et al., 2013) is a neural network trained on a simple task:

> **Given surrounding words, predict the centre word (CBOW), or given the centre word, predict the surrounding words (Skip-Gram).**

The network never uses the final prediction layer after training – only the **learned weight matrix** is kept. That weight matrix contains the word vectors.

### Why does predicting context produce good vectors?

The [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics) states:

> *"A word is characterised by the company it keeps."* – Firth (1957)

Words that appear in similar contexts will be pushed to similar positions in the vector space during training, because the network must use similar internal representations to make similar predictions.

### Skip-Gram Example

Sentence: `"The quick brown fox jumps"`  
Centre word: `fox`, Window size: 2

Training pairs:
```
(fox, The), (fox, quick), (fox, brown), (fox, jumps)
```

The network is trained to predict each context word given `fox`. After millions of such examples across a large corpus, `fox` ends up near other animal words, and action words near `jumps`.

### Training a Word2Vec model (Python)

```python
from gensim.models import Word2Vec

# Tokenised sentences
sentences = [
    ["the", "quick", "brown", "fox", "jumps"],
    ["dogs", "and", "cats", "are", "pets"],
    ["the", "cat", "sat", "on", "the", "mat"],
    ["a", "fox", "and", "a", "dog", "ran"],
]

# Train the model
model = Word2Vec(
    sentences,
    vector_size=50,   # Dimension of each word vector
    window=2,          # Context window size
    min_count=1,       # Ignore words with fewer than min_count occurrences
    workers=4,
    epochs=100,
)

# Get the vector for a word
fox_vector = model.wv["fox"]
print("fox vector shape:", fox_vector.shape)   # (50,)
print("fox vector:", fox_vector[:5], "...")    # First 5 dimensions

# Find the most similar words
print("\nWords most similar to 'fox':")
for word, score in model.wv.most_similar("fox"):
    print(f"  {word}: {score:.4f}")
```

---

## 5. GloVe

GloVe (Pennington et al., 2014) takes a different approach. Instead of training on individual context windows, it looks at **global word co-occurrence statistics** across the entire corpus.

### The Intuition

Build a co-occurrence matrix where entry `X[i][j]` counts how often word `i` appears near word `j` across all sentences:

|       | ice | steam | water | fashion |
|-------|-----|-------|-------|---------|
| solid | 1.9 | 0.1   | 1.2   | 0.05   |
| gas   | 0.1 | 2.1   | 1.1   | 0.08   |

The **ratio** `P(ice | solid) / P(steam | solid)` is large (solid relates to ice, not steam).  
The **ratio** `P(ice | gas)   / P(steam | gas)`  is small (gas relates to steam, not ice).

GloVe trains vectors such that the **dot product of two word vectors equals the log of their co-occurrence probability**. By capturing these ratios, GloVe vectors encode relationships like:

```
solid - gas ≈ ice - steam
```

### Using Pre-Trained GloVe Vectors

```python
import numpy as np

def load_glove(path):
    """Load GloVe vectors from a text file into a dictionary."""
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

# Download from https://nlp.stanford.edu/projects/glove/
# e.g. glove.6B.100d.txt
embeddings = load_glove("glove.6B.100d.txt")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare cat and dog
sim = cosine_similarity(embeddings["cat"], embeddings["dog"])
print(f"Similarity(cat, dog):   {sim:.4f}")   # High (~0.92)

sim2 = cosine_similarity(embeddings["cat"], embeddings["car"])
print(f"Similarity(cat, car):   {sim2:.4f}")  # Lower (~0.20)
```

---

## 6. Hands-On Code Examples

### Visualising Word Vectors with PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

sentences = [
    ["king", "queen", "man", "woman", "prince", "princess"],
    ["dog", "cat", "animal", "pet", "horse", "cow"],
    ["apple", "banana", "orange", "fruit", "mango"],
    ["python", "java", "code", "program", "software"],
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, epochs=200)

words = ["king", "queen", "man", "woman", "dog", "cat", "apple", "banana"]
vectors = np.array([model.wv[w] for w in words])

# Reduce to 2D for visualisation
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(coords[i, 0], coords[i, 1])
    plt.annotate(word, (coords[i, 0] + 0.02, coords[i, 1] + 0.02))

plt.title("Word Embeddings Visualised with PCA")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.tight_layout()
plt.savefig("word_vectors_pca.png", dpi=150)
plt.show()
```

### Word Analogy Test

```python
def word_analogy(model, word_a, word_b, word_c):
    """
    Solves: word_a is to word_b as word_c is to ???
    Formula: result ≈ word_b - word_a + word_c
    """
    result = model.wv.most_similar(
        positive=[word_b, word_c],
        negative=[word_a],
        topn=3,
    )
    return result

# Example: man is to king as woman is to ???
print(word_analogy(model, "man", "king", "woman"))
```

---

## Summary

| Method | Approach | Strength |
|--------|----------|----------|
| One-hot | Each word is a sparse vector | Simple, no information loss |
| Word2Vec | Predict context words | Fast, captures local context |
| GloVe | Factorize co-occurrence matrix | Captures global statistics |

Both Word2Vec and GloVe produce dense, meaningful vectors that power downstream NLP tasks such as text classification, machine translation, and question answering.

---

## Further Reading

- [Original Word2Vec paper (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [GloVe project page (Stanford NLP)](https://nlp.stanford.edu/projects/glove/)
- [Jay Alammar's visual guide to Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
