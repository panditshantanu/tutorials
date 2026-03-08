# Machine Learning Fundamentals

## What This Tutorial Covers

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Types of Machine Learning](#2-types-of-machine-learning)
3. [The Learning Loop: Loss, Gradient, and Update](#3-the-learning-loop-loss-gradient-and-update)
4. [Gradient Descent – Intuition and Code](#4-gradient-descent)
5. [Overfitting and Underfitting](#5-overfitting-and-underfitting)
6. [Bias–Variance Trade-off](#6-biasvariance-trade-off)
7. [Model Evaluation Metrics](#7-model-evaluation-metrics)

---

## 1. What Is Machine Learning?

Traditional programming:

```
Rules + Data → Output
```

Machine learning flips this:

```
Data + Output (labels) → Rules (model)
```

Instead of explicitly writing rules, we give the computer many examples and let it **discover the rules on its own** by adjusting internal parameters until its predictions match the labels.

---

## 2. Types of Machine Learning

### Supervised Learning
The dataset contains both inputs **X** and correct outputs **y**. The model learns a mapping `f: X → y`.

Examples: spam detection, house price prediction, image classification.

### Unsupervised Learning
Only inputs **X** are provided. The model discovers hidden structure.

Examples: customer segmentation (k-means clustering), anomaly detection, dimensionality reduction (PCA).

### Reinforcement Learning
An agent takes actions in an environment and receives rewards or penalties. It learns a policy that maximises cumulative reward.

Examples: game-playing agents (AlphaGo), robotics, recommendation systems.

---

## 3. The Learning Loop: Loss, Gradient, and Update

Every supervised learning algorithm follows this loop:

```
1. Make a prediction with current parameters.
2. Measure how wrong the prediction is  →  compute the Loss.
3. Calculate how each parameter contributed to the error  →  compute Gradients.
4. Nudge each parameter in the direction that reduces the Loss  →  Update.
5. Repeat.
```

### What is a Loss Function?

A loss function assigns a single number to how bad the model's predictions are. Lower loss = better predictions.

**Mean Squared Error (regression):**

```
MSE = (1/n) * Σ (y_pred - y_true)²
```

**Binary Cross-Entropy (classification):**

```
BCE = -(1/n) * Σ [ y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) ]
```

---

## 4. Gradient Descent

### The Intuition

Imagine you are blindfolded on a hilly landscape and want to reach the lowest valley (minimum loss). The only information you have is the slope under your feet. Gradient descent says:

> **Always take a small step in the direction that goes downhill.**

The **gradient** tells you the slope (direction of steepest ascent), so you move in the **opposite** direction:

```
parameter = parameter - learning_rate * gradient
```

The **learning rate** controls the step size.

- **Too large** a learning rate → overshoot the minimum, the loss bounces around.
- **Too small** a learning rate → takes forever to converge.

### Code: Gradient Descent from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: y = 2x + 1 + noise
np.random.seed(42)
X = np.random.rand(100) * 10
y = 2 * X + 1 + np.random.randn(100) * 1.5

# Initialise parameters
w = 0.0   # weight (slope)
b = 0.0   # bias (intercept)
lr = 0.001  # learning rate
epochs = 500

losses = []

for epoch in range(epochs):
    # Forward pass: predictions
    y_pred = w * X + b

    # Loss: Mean Squared Error
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # Gradients (partial derivatives of MSE w.r.t. w and b)
    dw = (2 / len(X)) * np.sum((y_pred - y) * X)
    db = (2 / len(X)) * np.sum(y_pred - y)

    # Parameter update
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w={w:.4f}, b={b:.4f}")

print(f"\nFinal parameters: w={w:.4f}, b={b:.4f}")
print(f"True parameters:  w=2.0000, b=1.0000")

# Plot the loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Curve During Training")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()
```

### Variants of Gradient Descent

| Variant | Description | Pro | Con |
|---------|-------------|-----|-----|
| Batch GD | Compute gradient on the full dataset | Stable convergence | Slow for large datasets |
| Stochastic GD (SGD) | Compute gradient on 1 sample | Fast updates | Noisy, unstable |
| Mini-Batch GD | Compute gradient on a small batch (e.g. 32 or 64 samples) | Balance of speed and stability | Requires tuning batch size |

---

## 5. Overfitting and Underfitting

### Underfitting

The model is **too simple** to capture the pattern in the data. It performs poorly on both training data and new data.

Visual sign: the model line is too flat or straight compared to the actual data trend.

### Overfitting

The model is **too complex** and memorises the training data, including noise. It performs very well on training data but poorly on new, unseen data.

Visual sign: the model line passes through every training point but curves wildly between them.

### The Sweet Spot

We want a model that:
- Learns the **true underlying pattern** from training data
- **Generalises** well to new examples

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

np.random.seed(0)
X_train = np.sort(np.random.rand(20)) * 6 - 3   # 20 points in [-3, 3]
y_train = np.sin(X_train) + np.random.randn(20) * 0.3

X_test = np.linspace(-3, 3, 200)
y_true = np.sin(X_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, degree, title in zip(axes, [1, 4, 15], ["Underfit (degree 1)",
                                                  "Good Fit (degree 4)",
                                                  "Overfit (degree 15)"]):
    coeffs = np.polyfit(X_train, y_train, degree)
    y_pred = np.polyval(coeffs, X_test)
    ax.scatter(X_train, y_train, color="black", zorder=5, label="Training data")
    ax.plot(X_test, y_true, "g--", label="True function")
    ax.plot(X_test, y_pred, "r-", label=f"Poly degree {degree}")
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("overfit_underfit.png", dpi=150)
plt.show()
```

---

## 6. Bias–Variance Trade-off

| | High Bias | High Variance |
|---|---|---|
| Other name | Underfitting | Overfitting |
| Training error | High | Low |
| Test error | High | High |
| Cause | Model too simple | Model too complex |
| Fix | More complex model, more features | More data, regularisation, simpler model |

**Total Error ≈ Bias² + Variance + Irreducible Noise**

Reducing bias typically increases variance, and vice versa. The goal is to find a model complexity that minimises total test error.

---

## 7. Model Evaluation Metrics

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5,  0.0, 2.0, 8.0])

print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

**When to use which metric:**

- **Accuracy** – balanced classes, overall correctness matters.
- **Precision** – false positives are costly (e.g. spam filtering: you don't want to delete real emails).
- **Recall** – false negatives are costly (e.g. cancer screening: you don't want to miss a diagnosis).
- **F1 Score** – you want a balance between precision and recall.

---

## Summary

```
Data  →  Model  →  Loss  →  Gradient  →  Update  →  (repeat)
```

The key ideas to remember:
- Gradient descent **optimises** model parameters by following the slope of the loss.
- The learning rate controls the step size.
- Overfitting = memorising; underfitting = too simple.
- Always evaluate on **held-out data** to measure true generalisation.

---

## Further Reading

- [Hands-On Machine Learning with Scikit-Learn & TensorFlow – Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [An Introduction to Statistical Learning – James et al.](https://www.statlearning.com/)
- [3Blue1Brown – Gradient Descent (YouTube)](https://www.youtube.com/watch?v=IHZwWFHWa-w)
