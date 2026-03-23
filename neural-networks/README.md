# Neural Networks – From Neurons to Deep Learning

## What This Tutorial Covers

1. [The Biological Inspiration](#1-the-biological-inspiration)
2. [A Single Neuron (Perceptron)](#2-a-single-neuron-perceptron)
3. [Activation Functions – Why Nonlinearity Matters](#3-activation-functions)
4. [Multi-Layer Networks – How They Represent Complex Patterns](#4-multi-layer-networks)
5. [Forward Pass – How a Prediction Is Made](#5-forward-pass)
6. [Backpropagation – How the Network Learns](#6-backpropagation)
7. [Building a Neural Network from Scratch](#7-building-a-neural-network-from-scratch)
8. [Training with PyTorch](#8-training-with-pytorch)

---

## 1. The Biological Inspiration

A biological neuron:
- Receives signals from other neurons via **dendrites**.
- Sums the incoming signals.
- If the sum exceeds a **threshold**, it **fires** (sends a signal) through the **axon** to the next neuron.

An artificial neuron mimics this:
- Receives numerical inputs `x₁, x₂, ...`.
- Multiplies each by a learnable **weight** `w₁, w₂, ...` and sums them up.
- Adds a **bias** term `b`.
- Passes the result through an **activation function** to decide the output.

```
output = activation( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )
```

---

## 2. A Single Neuron (Perceptron)

```python
import numpy as np

def perceptron(x, w, b, threshold=0.5):
    """
    x : input vector
    w : weight vector
    b : bias (scalar)
    """
    weighted_sum = np.dot(w, x) + b
    # Step activation: fire if weighted_sum > 0
    return 1 if weighted_sum > 0 else 0

# Example: AND gate
# Inputs: (x1, x2), Output: 1 only if both are 1
w = np.array([1.0, 1.0])
b = -1.5   # threshold

tests = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("AND Gate with a single perceptron:")
for x1, x2 in tests:
    out = perceptron(np.array([x1, x2]), w, b)
    print(f"  x1={x1}, x2={x2} → output={out}")
```

**Limitation:** A single perceptron can only separate data that is **linearly separable**. It cannot learn XOR without more layers.

---

## 3. Activation Functions

### Why Do We Need Them?

Without activation functions, stacking multiple linear layers is equivalent to a single linear layer:

```
Layer 2 (W₂) × Layer 1 (W₁ × x) = (W₂W₁) × x = W_single × x
```

No matter how many layers you add, the whole network collapses to a single linear transformation. Activation functions introduce **nonlinearity**, allowing networks to learn complex, curved decision boundaries.

### Common Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

def sigmoid(x):   return 1 / (1 + np.exp(-x))
def tanh(x):      return np.tanh(x)
def relu(x):      return np.maximum(0, x)
def leaky_relu(x, alpha=0.1): return np.where(x > 0, x, alpha * x)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, fn) in zip(axes, [
    ("Sigmoid", sigmoid),
    ("Tanh",    tanh),
    ("ReLU",    relu),
    ("Leaky ReLU", leaky_relu),
]):
    ax.plot(x, fn(x), linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("x")

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=150)
plt.show()
```

| Function | Range | Use Case | Problem |
|----------|-------|----------|---------|
| Sigmoid | (0, 1) | Output layer for binary classification | Vanishing gradients for deep nets |
| Tanh | (-1, 1) | Hidden layers (zero-centred) | Also suffers from vanishing gradients |
| ReLU | [0, ∞) | Default for hidden layers | "Dying ReLU" (neurons stuck at 0) |
| Leaky ReLU | (-∞, ∞) | Better alternative to ReLU | Requires tuning `alpha` |
| Softmax | (0, 1), sums to 1 | Output layer for multi-class classification | — |

---

## 4. Multi-Layer Networks

A **Multi-Layer Perceptron (MLP)** has:

1. **Input layer** – receives the raw features.
2. **Hidden layers** – learn intermediate representations.
3. **Output layer** – produces the final prediction.

```
Input layer       Hidden layer 1    Hidden layer 2    Output layer
  x₁ ──┐                                                 ŷ
  x₂ ──┼──→ [neurons] ──→ [neurons] ──→ [neurons] ──→   ŷ
  x₃ ──┘
```

**Key insight about hidden layers:**

Each hidden layer learns to detect different **features or patterns** in the data:

- Layer 1 might learn to detect edges (in images) or word patterns (in text).
- Layer 2 might combine edges into shapes, or words into phrases.
- Deeper layers combine lower-level features into higher-level concepts.

---

## 5. Forward Pass

The forward pass is how a network makes a prediction. It is just a series of matrix multiplications and activation functions:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Network architecture: 2 → 3 → 1
np.random.seed(0)

# Layer 1 weights and biases
W1 = np.random.randn(3, 2)   # 3 neurons, 2 inputs each
b1 = np.zeros((3, 1))

# Layer 2 weights and biases
W2 = np.random.randn(1, 3)   # 1 neuron, 3 inputs each
b2 = np.zeros((1, 1))

def forward(x):
    # x shape: (2, 1) – a single input sample
    z1 = W1 @ x + b1          # (3, 1)
    a1 = relu(z1)              # (3, 1) – hidden layer output
    z2 = W2 @ a1 + b2          # (1, 1)
    a2 = sigmoid(z2)           # (1, 1) – final prediction
    return a2, (z1, a1, z2, a2)

x = np.array([[0.5], [0.8]])   # Two input features
prediction, cache = forward(x)
print(f"Input: {x.ravel()}")
print(f"Prediction: {prediction[0, 0]:.4f}")
```

---

## 6. Backpropagation

Backpropagation is the algorithm that computes the **gradient of the loss with respect to every weight** in the network. It uses the **chain rule** from calculus.

### Intuition

When the network makes a wrong prediction:

1. Compute the loss.
2. Ask: "Which output neuron contributed most to this error?" → propagate error back to the output layer.
3. Ask: "Which hidden neurons caused the output neuron to be wrong?" → propagate error back to hidden layers.
4. Repeat until you reach the input layer.
5. Update all weights using gradient descent.

### Chain Rule in One Equation

```
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w
```

Each term asks a local question:
- `∂L/∂ŷ` – How does the loss change with the prediction?
- `∂ŷ/∂z` – How does the prediction change with the pre-activation value?
- `∂z/∂w` – How does the pre-activation value change with the weight?

---

## 7. Building a Neural Network from Scratch

```python
import numpy as np

class NeuralNetwork:
    """Two-layer neural network for binary classification."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # Xavier initialisation: helps gradients flow in deeper networks
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    # ── Activation functions ──────────────────────────────────────────────────
    def _relu(self, z):          return np.maximum(0, z)
    def _relu_grad(self, z):     return (z > 0).astype(float)
    def _sigmoid(self, z):       return 1 / (1 + np.exp(-z))

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, X):
        # X: (input_size, n_samples)
        self.Z1 = self.W1 @ X + self.b1          # (hidden_size, n_samples)
        self.A1 = self._relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2    # (output_size, n_samples)
        self.A2 = self._sigmoid(self.Z2)
        return self.A2

    # ── Loss ─────────────────────────────────────────────────────────────────
    def loss(self, A2, Y):
        m = Y.shape[1]
        # Binary cross-entropy, clipped for numerical stability
        A2 = np.clip(A2, 1e-9, 1 - 1e-9)
        return -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

    # ── Backward pass ─────────────────────────────────────────────────────────
    def backward(self, X, Y):
        m = X.shape[1]

        dZ2 = self.A2 - Y                                    # (output_size, m)
        dW2 = (dZ2 @ self.A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = self.W2.T @ dZ2                               # (hidden_size, m)
        dZ1 = dA1 * self._relu_grad(self.Z1)
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self, X, Y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            A2 = self.forward(X)
            l  = self.loss(A2, Y)
            self.backward(X, Y)
            if epoch % print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {l:.6f}")
        return self

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


# ── Train on XOR (not linearly separable) ─────────────────────────────────────
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])   # (2, 4)
Y = np.array([[0, 1, 1, 0]])   # XOR labels

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, lr=0.5)
nn.train(X, Y, epochs=5000, print_every=1000)

predictions = nn.predict(X)
print("\nXOR predictions:", predictions.ravel())
print("Expected:       ", Y.ravel())
print("Accuracy:", (predictions == Y).mean())
```

---

## 8. Training with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

# Define the model
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(2000):
    pred = model(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

with torch.no_grad():
    output = model(X)
    predicted = (output >= 0.5).float()
    print("\nPredictions:", predicted.squeeze().tolist())
    print("Expected:   ", Y.squeeze().tolist())
```

---

## Summary

```
Input → [W₁x + b₁] → Activation → [W₂x + b₂] → Activation → Output
                                                           ↑
                                           Backprop adjusts all weights
```

Key ideas:
- **Weights** are the learnable parameters.
- **Activation functions** add nonlinearity, enabling complex patterns.
- **Backpropagation + chain rule** computes gradients efficiently.
- **Gradient descent** updates the weights to reduce the loss.
- Deeper networks can learn hierarchical feature representations.

---

## Further Reading

- [3Blue1Brown – But what is a Neural Network? (YouTube)](https://www.youtube.com/watch?v=aircAruvnKk)
- [CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)](http://cs231n.github.io/)
- [Deep Learning book – Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
