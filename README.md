# Micrograd: Building an Autograd Engine from Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A minimal implementation of automatic differentiation and neural networks from first principles**

[Overview](#overview) • [Features](#features) • [Implementation Details](#implementation-details) • [Usage](#usage) • [Learning Outcomes](#learning-outcomes)

</div>

---

## 📚 Overview

This project is an educational implementation of **micrograd** - a minimal autograd engine that demonstrates the fundamental principles behind automatic differentiation and neural network training. Built from scratch using only Python and NumPy, this implementation provides deep insights into how modern deep learning frameworks like PyTorch and TensorFlow work under the hood.

> **Special Thanks**: This project is based on the excellent educational work by [Andrej Karpathy](https://github.com/karpathy) and his micrograd implementation. His insightful videos and explanations on building neural networks from scratch have been instrumental in understanding the fundamentals of automatic differentiation and backpropagation. Check out his [micrograd repository](https://github.com/karpathy/micrograd) and educational content for more deep learning insights!

### What is Autograd?

Automatic differentiation (autograd) is the technique that powers modern deep learning. Instead of manually computing derivatives, autograd engines automatically track operations and compute gradients using the chain rule, enabling efficient backpropagation through complex computational graphs.

---

## ✨ Features

### Core Components

- **`Value` Class**: A scalar value wrapper that tracks computation graphs

  - Automatic gradient computation via backpropagation
  - Support for basic operations: `+`, `-`, `*`, `/`, `**`, `tanh`, `exp`, `relu`
  - Topological sorting for efficient gradient flow

- **Neural Network Architecture**:

  - `Neuron`: Single neuron with weights, bias, and activation
  - `Layer`: Collection of neurons forming a layer
  - `MLP`: Multi-Layer Perceptron for building deep networks

- **Visualization**: Computation graph visualization using Graphviz

### Key Capabilities

✅ Forward and backward propagation  
✅ Gradient computation for all operations  
✅ Neural network training with gradient descent  
✅ Computation graph visualization  
✅ Binary classification with decision boundary visualization  
✅ Comparison with PyTorch implementation

---

## 🏗️ Implementation Details

### The Value Class

The heart of this implementation is the `Value` class, which wraps scalar values and tracks:

- **Data**: The actual numerical value
- **Gradient**: The derivative with respect to the output
- **Operation Graph**: Parent nodes and operations performed
- **Backward Function**: Local derivative computation

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
```

### Automatic Differentiation

The engine uses **reverse-mode automatic differentiation** (backpropagation):

1. **Forward Pass**: Build computation graph while performing operations
2. **Backward Pass**: Traverse graph in reverse topological order
3. **Chain Rule**: Multiply local derivatives by upstream gradients

### Example: Simple Computation

```python
# Define inputs
x1 = Value(2.0, label='x1')
w1 = Value(-3.0, label='w1')
b = Value(6.88, label='b')

# Forward pass
n = x1 * w1 + b
o = n.tanh()

# Backward pass
o.backward()

# Gradients are now computed!
print(x1.grad)  # ∂o/∂x1
print(w1.grad)  # ∂o/∂w1
print(b.grad)   # ∂o/∂b
```

---

## 🚀 Usage

### Prerequisites

```bash
pip install numpy matplotlib graphviz jupyter scikit-learn
```

**Note**: `scikit-learn` is required for the binary classifier notebook to generate the moons dataset.

### Running the Notebook

1. Clone the repository:

```bash
git clone https://github.com/Timalk16/micrograd-course.git
cd micrograd-course
```

2. Open the Jupyter notebooks:

```bash
jupyter notebook micrograd.ipynb
# or
jupyter notebook MLP_binary_classifier.ipynb
```

3. **`micrograd.ipynb`** - Run cells sequentially to:
   - Understand numerical differentiation
   - Build the `Value` class step by step
   - Visualize computation graphs
   - Train a simple neural network

4. **`MLP_binary_classifier.ipynb`** - Advanced example demonstrating:
   - Binary classification on the moons dataset
   - ReLU activation functions
   - SVM max-margin loss function
   - L2 regularization
   - Decision boundary visualization

### Training a Neural Network

**Basic Example** (`micrograd.ipynb`):

```python
# Create a multi-layer perceptron
mlp = MLP(3, [4, 4, 1])  # 3 inputs, hidden layers of 4, 4, output of 1

# Training data
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop
for epoch in range(20):
    # Forward pass
    ypred = [mlp(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

    # Backward pass
    for p in mlp.parameters():
        p.grad = 0.0
    loss.backward()

    # Update weights
    for p in mlp.parameters():
        p.data += -0.05 * p.grad

    print(f"Epoch {epoch}: Loss = {loss.data:.6f}")
```

**Binary Classification Example** (`MLP_binary_classifier.ipynb`):

```python
from sklearn.datasets import make_moons

# Generate non-linearly separable dataset
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert to -1 or 1

# Create MLP with ReLU activation
model = MLP(2, [16, 16, 1])  # 2 inputs, 2 hidden layers of 16, 1 output

# Training with SVM loss and L2 regularization
for epoch in range(100):
    # Forward pass
    inputs = [list(map(Value, xrow)) for xrow in X]
    scores = list(map(model, inputs))
    
    # SVM max-margin loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(y, scores)]
    data_loss = sum(losses) * (1 / len(losses))
    
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum([p*p for p in model.parameters()])
    total_loss = data_loss + reg_loss
    
    # Backward pass and update
    model.zero_grad()
    total_loss.backward()
    
    learning_rate = 1.0 - 0.9 * epoch / 100
    for p in model.parameters():
        p.data += -learning_rate * p.grad
    
    # Calculate accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)]
    print(f"Epoch {epoch}: Loss = {total_loss.data:.4f}, Accuracy = {sum(accuracy)/len(accuracy)*100:.1f}%")
```

---

## 🎓 Learning Outcomes

This project demonstrates understanding of:

### Mathematical Foundations

- **Calculus**: Derivatives, chain rule, partial derivatives
- **Numerical Methods**: Finite difference approximation
- **Linear Algebra**: Matrix operations, vector spaces

### Computer Science Concepts

- **Graph Theory**: Topological sorting, DAG traversal
- **Object-Oriented Design**: Class hierarchies, operator overloading
- **Algorithm Design**: Efficient gradient computation

### Deep Learning Principles

- **Backpropagation**: How gradients flow through networks
- **Neural Network Architecture**: Layers, neurons, activations
- **Optimization**: Gradient descent, parameter updates

### Software Engineering

- **Code Organization**: Modular design, reusable components
- **Visualization**: Graph representation and rendering
- **Testing**: Comparison with established frameworks (PyTorch)

---

## 📊 Project Structure

```
micrograd-course/
│
├── micrograd.ipynb              # Main implementation notebook
├── MLP_binary_classifier.ipynb   # Binary classification example
├── README.md                     # This file
│
└── Implementation Sections:
    ├── Numerical Differentiation
    ├── Value Class Definition
    ├── Operation Overloading
    ├── Backward Propagation
    ├── Graph Visualization
    ├── Neural Network Components
    ├── Training Example
    └── Binary Classification (MLP_binary_classifier.ipynb)
```

### Notebooks Overview

#### `micrograd.ipynb`
The foundational notebook that introduces the core concepts:
- Building the autograd engine from scratch
- Understanding automatic differentiation
- Implementing neural network components
- Basic training examples

#### `MLP_binary_classifier.ipynb`
An advanced practical example showcasing:
- **Binary Classification**: Training an MLP on the moons dataset (non-linearly separable data)
- **ReLU Activation**: Using Rectified Linear Units instead of tanh
- **SVM Loss**: Implementing max-margin loss for classification
- **Regularization**: L2 regularization to prevent overfitting
- **Visualization**: Decision boundary plotting to understand model behavior
- **Real-world Application**: Demonstrates how the micrograd engine can solve practical machine learning problems

---

## 🔬 Technical Highlights

### Gradient Computation

The engine correctly computes gradients for:

- **Addition**: `∂(a+b)/∂a = 1`, `∂(a+b)/∂b = 1`
- **Multiplication**: `∂(a*b)/∂a = b`, `∂(a*b)/∂b = a`
- **Power**: `∂(a^n)/∂a = n*a^(n-1)`
- **Tanh**: `∂tanh(a)/∂a = 1 - tanh²(a)`
- **ReLU**: `∂ReLU(a)/∂a = 1 if a > 0 else 0`
- **Exponential**: `∂exp(a)/∂a = exp(a)`

### Verification

The implementation is verified against PyTorch's autograd engine, producing identical gradient values (within floating-point precision).

---

## 🎯 Educational Value

This project is ideal for:

- **Students** learning deep learning fundamentals
- **Developers** wanting to understand autograd internals
- **Researchers** exploring gradient computation methods
- **Anyone** curious about how neural networks learn

---

## 📝 Notes

- This is an educational implementation focused on clarity over performance
- For production use, consider established frameworks like PyTorch or TensorFlow
- The implementation follows the micrograd approach created by Andrej Karpathy

---

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

This project would not have been possible without the incredible educational work of **Andrej Karpathy**. His micrograd project and detailed video explanations provided the foundation and inspiration for this implementation.

- **Andrej Karpathy** - Creator of micrograd and educator who makes deep learning accessible
  - [micrograd GitHub Repository](https://github.com/karpathy/micrograd)
  - [YouTube Channel](https://www.youtube.com/@AndrejKarpathy) - Excellent educational videos on neural networks and deep learning
  - His teaching style of building from first principles has been invaluable in understanding how modern deep learning frameworks work under the hood

Thank you, Andrej, for sharing your knowledge and making complex concepts approachable through hands-on implementation!

---

_Inspired by the micrograd implementation and educational content from the deep learning community, particularly the work on understanding neural networks from first principles._

---

<div align="center">

**Built with ❤️ for learning and understanding**

⭐ Star this repo if you find it helpful!

</div>
