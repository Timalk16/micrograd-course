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

### What is Autograd?

Automatic differentiation (autograd) is the technique that powers modern deep learning. Instead of manually computing derivatives, autograd engines automatically track operations and compute gradients using the chain rule, enabling efficient backpropagation through complex computational graphs.

---

## ✨ Features

### Core Components

- **`Value` Class**: A scalar value wrapper that tracks computation graphs
  - Automatic gradient computation via backpropagation
  - Support for basic operations: `+`, `-`, `*`, `/`, `**`, `tanh`, `exp`
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
pip install numpy matplotlib graphviz jupyter
```

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/Timalk16/micrograd-course.git
cd micrograd-course
```

2. Open the Jupyter notebook:
```bash
jupyter notebook micrograd.ipynb
```

3. Run cells sequentially to:
   - Understand numerical differentiation
   - Build the `Value` class step by step
   - Visualize computation graphs
   - Train a simple neural network

### Training a Neural Network

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
├── micrograd.ipynb          # Main implementation notebook
├── README.md                # This file
│
└── Implementation Sections:
    ├── Numerical Differentiation
    ├── Value Class Definition
    ├── Operation Overloading
    ├── Backward Propagation
    ├── Graph Visualization
    ├── Neural Network Components
    └── Training Example
```

---

## 🔬 Technical Highlights

### Gradient Computation

The engine correctly computes gradients for:
- **Addition**: `∂(a+b)/∂a = 1`, `∂(a+b)/∂b = 1`
- **Multiplication**: `∂(a*b)/∂a = b`, `∂(a*b)/∂b = a`
- **Power**: `∂(a^n)/∂a = n*a^(n-1)`
- **Tanh**: `∂tanh(a)/∂a = 1 - tanh²(a)`
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
- The implementation follows the micrograd approach popularized by Andrej Karpathy

---

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

Inspired by the micrograd implementation and educational content from the deep learning community, particularly the work on understanding neural networks from first principles.

---

<div align="center">

**Built with ❤️ for learning and understanding**

⭐ Star this repo if you find it helpful!

</div>

