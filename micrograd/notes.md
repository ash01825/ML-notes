## 1. Theory, Math & Derivations

### Concepts: Rigorous Definitions

| Term | Definition | Context/Source
| :--- | :--- | :---
| **Autograd** | Short for automatic gradient, this engine computes the gradient of a mathematical expression automatically. Micrograd is explicitly a **scalar-valued** Autograd system. |
| **Backpropagation** | An efficient algorithm that computes the partial derivatives ($\nabla L$) of a scalar loss function ($L$) with respect to all internal parameters ($W$) in a directed acyclic computation graph. It achieves this by recursively applying the Chain Rule backwards from the output node. |
| **Derivative** | The instantaneous rate of change or slope of a function at a specific point. It quantifies the sensitivity with which a function's output responds to a tiny perturbation ($h$) in its input. |
| **Chain Rule** | A calculus theorem stating that the derivative of a composite function is the product of the derivatives of the functions involved. For $z$ depending on $y$, and $y$ depending on $x$: $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$. |
| **Loss Function ($L$)** | A single scalar value that quantifies the total performance of the neural network across a dataset. Training aims to minimize this value. |
| **Gradient Descent** | An iterative optimization procedure where network parameters ($p$) are updated by taking small steps ($\eta$) in the direction **opposite** to the loss gradient ($\nabla L$), thus minimizing the loss function $L$. |

### Formulas, Gradients, and Step-by-Step Calculus

All variables ($x, y, L$) in Micrograd are treated as scalars ($\in \mathbb{R}$), meaning their associated matrix dimensions are $1 \times 1$.

#### A. Fundamental Derivative Definition
The numerical approximation of the derivative $\frac{df}{dx}$ for a small step size $h$ is defined as the rise over run:
$$ \frac{df}{dx} \approx \frac{f(x+h) - f(x)}{h} \quad $$

#### B. Local Derivatives and Backpropagation Rules

In the context of backpropagation, the gradient of the final loss $L$ with respect to an input variable $P$ is calculated by multiplying the known incoming gradient $\frac{dL}{dQ}$ by the local derivative $\frac{dQ}{dP}$.

| Operation | Output $Q$ | Local Derivative $\frac{dQ}{dP}$ | Backprop Rule ($\frac{dL}{dP}$ accumulation) | Source
| :--- | :--- | :--- | :--- | :---
| **Addition** | $Q = P_1 + P_2$ | $\frac{\partial Q}{\partial P_1} = 1.0$ | $\frac{dL}{dP_1} += \frac{dL}{dQ} \cdot 1.0$. Gradient is routed (distributed). |
| **Multiplication** | $Q = P_1 \cdot P_2$ | $\frac{\partial Q}{\partial P_1} = P_2$ | $\frac{dL}{dP_1} += \frac{dL}{dQ} \cdot P_2$. Gradient is scaled by the *other* input's forward value. |
| **Power (Constant $k$)** | $Q = P^k$ | $\frac{dQ}{dP} = k \cdot P^{k-1}$ | $\frac{dL}{dP} += \frac{dL}{dQ} \cdot (k \cdot P^{k-1})$. Applies the power rule. |
| **Exponentiation** | $Q = e^P$ | $\frac{dQ}{dP} = e^P = Q$ | $\frac{dL}{dP} += \frac{dL}{dQ} \cdot Q$. The local derivative is the output value itself. |

#### C. Tanh Activation Function Gradient Derivation

The $\tanh$ function is defined as $O = \tanh(N)$.
The derivative $\frac{dO}{dN}$ is needed for the local gradient.

1.  **Target Derivative:** We seek $\frac{d}{dN} \tanh(N)$.
2.  **Known Identity:** $\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$.
3.  **Local Derivative Result:** $\frac{dO}{dN} = 1 - O^2$.
4.  **Backpropagation Step:** The gradient $\frac{dL}{dN}$ is accumulated using the Chain Rule:
    $$ \frac{dL}{dN} += \frac{dL}{dO} \cdot (1 - O^2) $$
    In the code, where `out` is $O$ and `self` is $N$: `self.grad += out.grad * (1.0 - out.data**2)`.

#### D. Mean Squared Error (MSE) Loss

The loss $L$ over $N$ examples is the sum of squared differences between predictions ($y_{pred, i}$) and targets ($y_{gt, i}$).
$$ L = \sum_{i=1}^{N} (y_{pred, i} - y_{gt, i})^2 \quad $$

### Why: Explaining Design Choices

1.  **Scalar-Valued Engine:** The decision to build Micrograd as a scalar-valued engine (working with individual numbers like $-4$ and $2$) was purely **pedagogical**. This choice simplifies the implementation by allowing focus on the Chain Rule and backpropagation without the complexity of N-dimensional tensors (arrays) used for efficiency in production environments.
2.  **Gradient Accumulation (`+=`):** In multivariate calculus, when a single variable ($A$) contributes to the final output ($L$) through multiple paths (e.g., $A \to D$ and $A \to E$, both of which feed into $L$), the total derivative $\frac{dL}{dA}$ must be the sum of the derivatives from each path. Using **`p.grad += ...`** instead of `p.grad = ...` ensures this critical accumulation, preventing gradients from being overwritten by subsequent backpropagation steps.
3.  **Weight Initialization:** Neuron weights and biases are initialized randomly (e.g., between $-1$ and $1$). This breaks symmetry and ensures different neurons learn different features, as starting them all at the same value would lead to identical gradient updates.
4.  **Topological Sort:** The expression graph (DAG) must be traversed in a specific order during backpropagation. **Topological sort** guarantees that a node's `_backward` function is only executed after all subsequent nodes that depend on it have already propagated their full gradients to that node. This is essential for correctness.

---

## 2. Code, Tensors & Internals

### Line-by-Line Analysis: The `Value` Object

The `Value` class wraps a scalar float and maintains the graph structure and gradient information.

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0 # dL/d(self.data), initialized to zero (no effect)
        self._prev = set(_children) # Set of parent Value objects for graph traversal
        self._op = _op # String identifying the operation ('+', '*', 'tanh')
        self._backward = lambda: None # Placeholder for local gradient function
```

#### Tensor Tracking and PyTorch Nuance

Micrograd explicitly works with $1 \times 1$ scalar values ($\in \mathbb{R}$). PyTorch, in contrast, uses **tensors**, which are $N$-dimensional arrays of scalars, primarily for computational efficiency via parallel processing.

When replicating Micrograd's behavior in PyTorch:
1.  **Tensor Initialization:** Scalars are wrapped into a $1 \times 1$ tensor (e.g., `torch.tensor([2.0], ...)`).
2.  **Data Type:** The data type is typically cast to `torch.double` (Float64) to match Python's default precision, as PyTorch defaults to `float32`.
3.  **Gradient Requirement:** Leaf nodes must be explicitly declared with `requires_grad=True` to signal that their gradient must be tracked for optimization.
4.  **Extraction:** The scalar value of a $1 \times 1$ tensor is retrieved using `.item()`.
5.  **Performance:** While Micrograd demonstrates the math, PyTorch leverages large tensor operations to utilize the parallelism of hardware (e.g., GPUs), drastically increasing training speed; the mathematical core (Chain Rule/Backprop) remains identical.

### Internals: Special Methods and Closures

| Method/Implementation | PyTorch Equivalent | Nuance and Purpose
| :--- | :--- | :---
| **`__add__`, `__mul__`** | Standard `+`, `*` operators on tensors. | Define how two `Value` objects interact. Crucially, they contain logic to **wrap non-Value inputs** (e.g., `a + 1`) into `Value` objects to maintain computational graph integrity.
| **`__rmul__`, `__radd__`** | Python fallback mechanism for commutative operations. | These are called when the `Value` object is on the right (e.g., `2 * a`). Python checks if the left operand (`2`) can handle the multiplication; if not, it checks the right operand (`a`) for an `__rmul__` definition, which swaps the operands (`a.__mul__(2)`).
| **`_backward` (Closure)** | `backward` method of PyTorch's `Function` (custom autograd classes). | A Python **closure** function defined within the arithmetic methods (e.g., `__add__`). This function captures and stores the *local derivatives* and references to the input/output nodes at the time of the forward pass, ensuring the correct chain rule calculation during the backward pass.
| **`backward()`** | `loss.backward()` | The root function that orchestrates the entire backpropagation process: 1. Initializes the root gradient to 1.0. 2. Builds the `topo` list (topological sort). 3. Iterates through the `topo` list in reverse order, executing each node's `_backward` closure.

---

## 3. "The Laboratory": Pitfalls & Visuals

### Traps, Failure Mechanisms, and Fixes

| Trap | Failure Mechanism/Physical Failure | Specific Fix
| :--- | :--- | :---
| **Gradient Overwriting (Multi-Use Variables)** | A shared input variable (e.g., $A$ in $B=A+A$) receives gradients from multiple branches, but sequential execution of `_backward` uses assignment (`=`) instead of accumulation, leading to the gradient from one branch overwriting contributions from others. | **Fix:** Implement **accumulation** (`+=`) for all gradient assignments within all `_backward` functions. This adheres to the generalized multivariate Chain Rule, ensuring contributions are summed.
| **Forgetting to Zero Gradients** | The accumulated gradients (`p.grad`) are not reset between optimization steps. Because backpropagation always uses `+=`, the gradients from the new loss calculation pile onto the old gradients, leading to an incorrect, massive, and unstable effective step size. | **Fix:** Implement a **Zero Grad** loop before every backward pass: `for p in n.parameters(): p.grad = 0.0`.

### Visuals: Computation Graph and Loss Charts

*   **Computation Graph Visualization (`drawdot`):**
    *   **Square Nodes:** Represent `Value` objects, displaying both the computed value (`data`) and the accumulated loss gradient ($\frac{dL}{d(\text{Node})}$, or `grad`).
    *   **Circular Operation Nodes:** Faux nodes (e.g., '+', '*') inserted by the visualization utility (Graphviz) to clearly denote the mathematical operation connecting the `Value` objects.
    *   **Edges:** Show the forward data flow, implicitly indicating the reverse flow of gradient information.
*   **Loss/Optimization Charts:**
    *   **Decreasing Loss:** Indicates successful training, where predictions are aligning with targets.
    *   **Learning Rate Stability:** A low learning rate ($\eta$) results in slow, controlled descent. A high $\eta$ can cause the loss to become **unstable** or **explode** (increase suddenly) because the optimization steps are too large, violating the local approximation of the gradient.

### Magic Numbers (Hyperparameters)

| Hyperparameter | Value(s) | Logic/Rationale
| :--- | :--- | :---
| **Initialization Range** | $\text{W} \sim [-1, 1]$ | Provides small, non-zero starting weights for the neurons.
| **Learning Rate ($\eta$ / Step Size)** | $0.01, 0.05, 0.1$. | Controls how aggressively parameters are adjusted in the direction of the negative gradient. Tuning is an "subtle art": too high leads to instability/overshooting; too low leads to slow convergence.
| **Numerical Step ($h$)** | $0.001$. | Used for gradient checking (numerical verification). Must be small to converge toward the true derivative, but not so small as to introduce floating-point precision errors.

---

## 4. Literature Connections

The design choices in Micrograd directly mirror generalized mechanisms found in deep learning libraries and research papers:

*   **Activation Functions:** The use of $\tanh$ (and the mention of $\text{ReLU}$ and $\text{sigmoid}$) connects to the necessity of introducing non-linearity in multi-layer perceptrons, preventing the entire network from collapsing into a single linear function.
*   **Module API:** The definition of `Neuron`, `Layer`, and `MLP` classes, along with the `parameters()` method and the `zero_grad()` implementation, explicitly mirrors the structure and Application Programming Interface (API) defined by the base `nn.Module` class in PyTorch.
*   **Stochastic Gradient Descent (SGD):** The iterative process of forward pass, backward pass, and parameter update ($\mathbf{p} \leftarrow \mathbf{p} - \eta \cdot \nabla L$) is the fundamental mechanism of SGD. Although basic, it underpins more advanced optimizers.
*   **Loss Functions:** While MSE loss is used for simplicity, the sources reference alternatives like **Cross-Entropy Loss** (common for classification, e.g., predicting the next token in GPT models) and **Max Margin Loss**.
*   **Regularization and Scheduling:** Concepts like **L2 regularization** (controlling overfitting) and **Learning Rate Decay** (shrinking $\eta$ over time for better fine-tuning) are mentioned as necessary additions for handling complex, larger datasets.

---

**Analogy:**

Micrograd operates like a meticulous accountant tracking expenses across a business built from small contracts (the operations). The **forward pass** is calculating the final profit ($L$). **Backpropagation** is the auditor starting at the final profit and recursively working backwards through every ledger entry (local derivative) to find out how much each initial investment (weight $W$) contributed to—or detracted from—that final profit. The **Chain Rule** is the required mathematical formula ensuring the contribution rate from each contract is correctly factored into the next. The **Gradient Descent** step is then adjusting those initial investments (weights) inversely to their calculated contribution to loss, aiming to maximize eventual profit (minimize loss).