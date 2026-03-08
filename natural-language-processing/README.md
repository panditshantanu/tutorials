# Natural Language Processing (NLP) – Overview

## What This Tutorial Covers

1. [What is NLP?](#1-what-is-nlp)
2. [Text Preprocessing Pipeline](#2-text-preprocessing-pipeline)
3. [Tokenisation](#3-tokenisation)
4. [Bag of Words and TF-IDF](#4-bag-of-words-and-tf-idf)
5. [Language Models – From N-grams to Transformers](#5-language-models)
6. [Hands-On: Text Classification with scikit-learn](#6-hands-on-text-classification)

---

## 1. What Is NLP?

**Natural Language Processing** is the branch of AI that enables computers to understand, interpret, and generate human language.

Applications include:
- Spam detection
- Sentiment analysis (positive / negative review)
- Machine translation (English → French)
- Question answering (ChatGPT, Google)
- Named entity recognition (finding people, places, dates in text)

The core challenge: human language is **ambiguous, context-dependent, and evolving**.

```
"I saw the man with the telescope."
```

Does this mean:
- I used a telescope to see the man? OR
- I saw a man who was holding a telescope?

Humans resolve this from context. Machines must learn to do the same.

---

## 2. Text Preprocessing Pipeline

Raw text must be cleaned and standardised before a model can learn from it.

```
Raw text
  → Lowercase
  → Remove punctuation / special characters
  → Tokenise (split into words or sub-words)
  → Remove stop words (optional)
  → Stem or Lemmatise (optional)
  → Convert to numbers (vectorisation)
```

```python
import re
import string

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

sample = "Hello, World! This is an NLP Tutorial -- very exciting!!!"
print(preprocess(sample))
# hello world this is an nlp tutorial very exciting
```

---

## 3. Tokenisation

Tokenisation splits a text string into a list of smaller units called **tokens** (usually words or sub-words).

### Word Tokenisation

```python
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt", quiet=True)

text = "Natural language processing enables computers to understand human language."
tokens = word_tokenize(text)
print(tokens)
# ['Natural', 'language', 'processing', 'enables', 'computers',
#  'to', 'understand', 'human', 'language', '.']
```

### Stop Word Removal

Stop words (e.g. "the", "is", "to") are very common and usually carry little meaning for classification tasks.

```python
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
filtered = [t for t in tokens if t.lower() not in stop_words and t.isalpha()]
print(filtered)
# ['Natural', 'language', 'processing', 'enables', 'computers', 'understand', 'human', 'language']
```

### Lemmatisation

Reduces words to their **dictionary base form**: "running" → "run", "better" → "good".

```python
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

words = ["running", "ran", "runs", "better", "dogs", "studies"]
lemmatised = [lemmatizer.lemmatize(w) for w in words]
print(lemmatised)
# ['running', 'ran', 'run', 'better', 'dog', 'study']
```

---

## 4. Bag of Words and TF-IDF

### Bag of Words (BoW)

BoW represents a document as a **vector of word counts**, ignoring order.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "I love deep learning",
    "Deep learning is part of machine learning",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("\nBoW matrix:\n", X.toarray())
```

**Problem with BoW:** Common words like "the", "is" get high counts but carry little information. A word that appears in every document is not useful for distinguishing documents.

### TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF down-weights words that appear frequently across many documents and up-weights words that are distinctive to a specific document.

```
TF(t, d)  = (number of times t appears in d) / (total words in d)
IDF(t)    = log( total documents / documents containing t )
TF-IDF    = TF × IDF
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

import pandas as pd
df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf.get_feature_names_out(),
    index=["doc1", "doc2", "doc3"],
)
print(df.round(3))
```

**Interpretation:** A word with a high TF-IDF score in a document is important *to that document* but not common across the whole corpus.

---

## 5. Language Models

A **language model** assigns a probability to a sequence of words:

```
P("The cat sat on the mat") = ?
```

More useful in practice: given a context, what is the most likely next word?

### N-gram Language Model

An n-gram model estimates the probability of the next word using the previous `n-1` words.

```python
from collections import defaultdict, Counter
import random

def build_bigram_model(sentences):
    """Build a bigram language model."""
    model = defaultdict(Counter)
    for sentence in sentences:
        tokens = sentence.lower().split()
        for w1, w2 in zip(tokens, tokens[1:]):
            model[w1][w2] += 1
    return model

def generate_text(model, start_word, max_words=10):
    current = start_word
    words = [current]
    for _ in range(max_words - 1):
        if current not in model:
            break
        next_words = list(model[current].keys())
        counts     = list(model[current].values())
        total      = sum(counts)
        probs      = [c / total for c in counts]
        current    = random.choices(next_words, weights=probs)[0]
        words.append(current)
    return " ".join(words)

corpus = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "the rat sat on the mat",
    "the dog sat on the floor",
]

model = build_bigram_model(corpus)
random.seed(42)
print(generate_text(model, "the"))
```

**Limitations of N-gram models:**
- Cannot capture long-range dependencies.
- Vocabulary explosion as n grows.

Modern language models (GPT, BERT, T5) use **Transformer architectures** and self-attention to capture dependencies across entire sequences. See the [neural-networks tutorial](../neural-networks/README.md) for the foundations needed to understand Transformers.

---

## 6. Hands-On: Text Classification

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Use 4 categories from the 20 Newsgroups dataset
categories = [
    "sci.space",
    "rec.sport.hockey",
    "talk.politics.guns",
    "comp.graphics",
]

train = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
test  = fetch_20newsgroups(subset="test",  categories=categories, remove=("headers", "footers", "quotes"))

# Build a simple TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
    ("clf",   LogisticRegression(max_iter=1000)),
])

pipeline.fit(train.data, train.target)
y_pred = pipeline.predict(test.data)

print(classification_report(test.target, y_pred, target_names=categories))
```

**Expected output:** accuracy ~90%+ with just TF-IDF features and logistic regression, demonstrating how powerful even simple NLP pipelines can be.

---

## Summary

```
Raw Text
  → Preprocessing (lower, clean)
  → Tokenisation
  → Vectorisation (BoW / TF-IDF / Embeddings)
  → Model (Logistic Regression / Neural Network / Transformer)
  → Prediction
```

| Representation | Captures Order | Captures Meaning | Dimensionality |
|----------------|---------------|-----------------|----------------|
| BoW            | No            | No              | Vocabulary size |
| TF-IDF         | No            | Partially       | Vocabulary size |
| Word Embeddings| No (word-level)| Yes             | 50–300         |
| Transformers   | Yes           | Yes             | 768–1024+      |

---

## Further Reading

- [Speech and Language Processing – Jurafsky & Martin (free online)](https://web.stanford.edu/~jurafsky/slp3/)
- [The Illustrated BERT (Jay Alammar)](https://jalammar.github.io/illustrated-bert/)
- [fast.ai NLP Course](https://course.fast.ai/)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
