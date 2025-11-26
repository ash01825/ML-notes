This guide provides a dense, rigorous examination of the character-level bigram language model, its two implementations (statistical and neural), and the underlying mathematical and engineering principles drawn from the sources.

## 1. Theory, Math & Derivations

### Concepts: Rigorous Definitions

| Term | Rigorous Definition and Context |
| :--- | :--- |
| **Character-Level Language Model** | A generative model structured to predict the probability of the next character ($c_{t+1}$) given a sequence of preceding characters ($c_1, c_2, \dots, c_t$). |
| **Bigram Language Model** | A specific character-level language model where the probability of the next character $c_{t+1}$ is conditioned exclusively on the immediately preceding character $c_t$: $P(c_{t+1} \,|\, c_t)$. |
| **Likelihood ($\mathcal{L}$)** | The joint probability of observing the entire dataset ($\mathcal{D}$) given a set of model parameters ($\mathbf{\Theta}$). Calculated as the product of the probabilities assigned by the model to every observed bigram instance in the training data. The primary objective of training is Maximum Likelihood Estimation (MLE): $\max_{\mathbf{\Theta}} \mathcal{L}(\mathcal{D} \,|\, \mathbf{\Theta})$. |
| **Log Likelihood ($\mathcal{LL}$)** | The logarithm of the likelihood, $\mathcal{LL} = \log(\mathcal{L})$. Used for numerical stability as $\mathcal{L}$ is typically extremely small (product of many numbers $< 1$) and converts the product operation into a summation: $\mathcal{LL} = \sum \log(P(c_{t+1}|c_t))$. |
| **Negative Log Likelihood (NLL)** | The quantity $-\mathcal{LL}$. Used as a loss function where the minimization objective is equivalent to maximizing the likelihood. NLL measures the penalty assigned when the model makes a prediction, where lower NLL is better. |
| **Average Negative Log Likelihood (Loss)** | $\mathcal{L}_{NLL} = -\frac{1}{B} \sum_{i=1}^{B} \log(P_i)$, where $B$ is the number of examples. This loss serves as the standard performance metric, equivalent to the **Cross-Entropy Loss** in classification, to be minimized via optimization. |
| **Logits ($\mathbf{Z}$)** | The raw, unnormalized outputs of a linear neural network layer before the Softmax transformation is applied. These outputs $\mathbf{Z} \in \mathbb{R}$ range across the entire real line and are interpreted as **log counts** by the model. |
| **Softmax** | A normalization function applied to logits that transforms them into a valid probability distribution. It involves element-wise exponentiation followed by normalization, ensuring all resulting probabilities are positive and sum to one. |
| **L2 Regularization** | An augmentation to the loss function ($\mathcal{L}_{Total} = \mathcal{L}_{NLL} + \mathcal{L}_{Reg}$) designed to penalize large magnitudes of the weight parameters $\mathbf{W}$. Mathematically equivalent to adding uniform pseudo-counts (model smoothing) to prevent highly peaked, zero-probability distributions. |

### Formulas and Derivations (LaTeX)

Let $V$ be the vocabulary size ($V=27$).
Let $B$ be the batch size or the total number of examples ($B \approx 228,000$ for full training).
Let $\mathbf{X}_{enc} \in \mathbb{R}^{B \times V}$ be the batch of one-hot encoded inputs.
Let $\mathbf{W} \in \mathbb{R}^{V \times V}$ be the weight matrix (model parameters).
Let $\mathbf{Y} \in \mathbb{Z}^{B}$ be the target indices.

#### Forward Pass: Logits to Probabilities

**1. Linear Transformation (Logits $\mathbf{Z}$):**
The input batch $\mathbf{X}_{enc}$ is multiplied by the weight matrix $\mathbf{W}$ (no bias term is present in this simple layer).

$$
\mathbf{Z} = \mathbf{X}_{enc} \mathbf{W} \quad \text{where } \mathbf{Z} \in \mathbb{R}^{B \times V}
$$

**2. Softmax (Probabilities $\mathbf{P}$):**
The logits are converted to a probability matrix $\mathbf{P}$ via the Softmax function (exponentiate and normalize).

$$
\mathbf{P}_{i, j} = \frac{\exp(\mathbf{Z}_{i, j})}{\sum_{k=1}^{V} \exp(\mathbf{Z}_{i, k})} \quad \text{where } \mathbf{P} \in \mathbb{R}^{B \times V}
$$

#### Loss Calculation

**3. Negative Log Likelihood (NLL) Loss ($\mathcal{L}_{NLL}$):**
The loss is calculated by selecting the probabilities assigned to the true targets ($\mathbf{P}_{i, Y_i}$), taking the negative logarithm, and averaging over the batch $B$.

$$
\mathcal{L}_{NLL} = - \frac{1}{B} \sum_{i=1}^{B} \log(\mathbf{P}_{i, Y_i})
$$

**4. L2 Regularization Loss ($\mathcal{L}_{Reg}$):**
$\lambda$ is the regularization strength (e.g., $0.01$).

$$
\mathcal{L}_{Reg} = \lambda \cdot \text{Mean}(\mathbf{W}^2) = \frac{\lambda}{V^2} \sum_{i=1}^{V} \sum_{j=1}^{V} \mathbf{W}_{i, j}^2
$$

**5. Total Loss ($\mathcal{L}_{Total}$):**
$$
\mathcal{L}_{Total} = \mathcal{L}_{NLL} + \mathcal{L}_{Reg}
$$

### Why: Design Rationale

The structure $Z = X_{enc}W$ serves a critical dual purpose:

1.  **Computational Efficiency:** Matrix multiplication efficiently calculates the dot product for $B$ inputs across $V$ neurons simultaneously.
2.  **Lookup Equivalence:** Because $\mathbf{X}_{enc}$ is one-hot encoded, $\mathbf{X}_{enc} \mathbf{W}$ functionally acts as a lookup table. If the input index $i$ has a '1' at position $k$, the operation $\mathbf{X}_{enc} \mathbf{W}$ explicitly plucks out the $k$-th row of $\mathbf{W}$.
    *   **Reason:** This confirms that the initial neural network parameters $\mathbf{W}$ literally represent the **log counts** of the bigram frequencies. The row index $k$ (the input character) maps directly to the $k$-th row of $\mathbf{W}$ (the logits/log probabilities for the next character).

### Gradients (Differentiable Operations)

All steps from $\mathbf{X}_{enc}$ to $\mathcal{L}_{Total}$ are differentiable operations (multiplication, exponentiation, summation, division, logarithm). PyTorch's `loss.backward()` automatically computes the gradients $\frac{\partial \mathcal{L}_{Total}}{\partial \mathbf{W}}$.

**Gradient Calculation Chain (Conceptual, based on derived structure):**

1.  **Gradient of NLL w.r.t. Logits ($\mathbf{Z}$):**
    The gradient flowing backward from the Softmax/NLL boundary is calculated as the error signal, $\mathbf{P} - \mathbf{Y}_{one\_hot}$.
    $$
    \frac{\partial \mathcal{L}_{NLL}}{\partial \mathbf{Z}} = \mathbf{P} - \mathbf{Y}_{one\_hot} \quad \text{where } \mathbf{Y}_{one\_hot} \in \mathbb{R}^{B \times V} \text{ (one-hot targets)}
    $$

2.  **Gradient of NLL w.r.t. Weights ($\mathbf{W}$):**
    Using the chain rule and the derivative of matrix multiplication:
    $$
    \frac{\partial \mathcal{L}_{NLL}}{\partial \mathbf{W}} = \mathbf{X}_{enc}^T \left( \frac{\partial \mathcal{L}_{NLL}}{\partial \mathbf{Z}} \right)
    $$

3.  **Gradient of Regularization w.r.t. Weights ($\mathbf{W}$):**
    $$
    \frac{\partial \mathcal{L}_{Reg}}{\partial \mathbf{W}} = \frac{2 \lambda}{V^2} \mathbf{W}
    $$

4.  **Weight Update (Gradient Descent):**
    The weights are updated iteratively opposite the total gradient direction using a learning rate $\eta$ (e.g., $0.5$ or $50$).
    $$
    \mathbf{W} \leftarrow \mathbf{W} - \eta \left( \frac{\partial \mathcal{L}_{NLL}}{\partial \mathbf{W}} + \frac{\partial \mathcal{L}_{Reg}}{\partial \mathbf{W}} \right)
    $$

## 2. Code, Tensors & Internals

### Tensor Tracking and Transformation

Assuming $V=27$ and working with a batch size $B=5$ (the "Emma" example):

| Variable | Creation/Transformation | Input Shape | Output Shape | $Dtype$ (Crucial Detail) |
| :--- | :--- | :--- | :--- | :--- |
| $\mathbf{X}, \mathbf{Y}$ (Raw Indices) | `torch.tensor(xs)` | $$ | $$ | `torch.int64` (inferred) |
| $\mathbf{X}_{enc}$ (One-Hot Int) | `F.one_hot(X, 27)` | $$ | $$ | `torch.int64` (Default output) |
| $\mathbf{X}_{enc}$ (One-Hot Float) | `.float()` | $$ | $$ | **`torch.float32`** (Required for NN arithmetic) |
| $\mathbf{W}$ (Weights) | `torch.randn(27, 27, req_grad=True)` | N/A | $$ | `torch.float32` (Leaf tensor) |
| $\mathbf{Z}$ (Logits) | $\mathbf{X}_{enc} @ \mathbf{W}$ | $ \cdot$ | $$ | `torch.float32` (Intermediate) |
| $\mathbf{C}$ (Counts/Exp) | $\mathbf{Z}.exp()$ | $$ | $$ | `torch.float32` (Intermediate) |
| $\mathbf{S}$ (Sums) | $\mathbf{C}.sum(dim=1, \text{keepdim=True})$ | $$ | **$$** | `torch.float32` (Crucial for broadcasting) |
| $\mathbf{P}$ (Probabilities) | $\mathbf{C} / \mathbf{S}$ | $ /$ | $$ | `torch.float32` (Output distribution) |
| $\mathbf{P}_{target}$ | $\mathbf{P}[\text{range}(5), \mathbf{Y}]$ | $$ indexed by 2 vectors | $$ | `torch.float32` (Selected probabilities) |

### Nuance: PyTorch Specifics

1.  **Broadcasting Semantics (Division):** The division $\mathbf{C} / \mathbf{S}$ relies on PyTorch's broadcasting rules. The shape $$ divided by $$ is allowed because dimensions are either equal (dim 0: 5=5) or one dimension is 1 (dim 1: 27 vs 1). The $$ tensor is conceptually replicated 27 times horizontally to match the $$ shape, allowing for element-wise row normalization.
2.  **`requires_grad=True`:** Must be explicitly set during the initialization of the parameter tensor $\mathbf{W}$ so that PyTorch tracks all subsequent operations to construct the computation graph necessary for backpropagation.
3.  **In-Place Update (`.data`):** The update `W.data += -eta * W.grad` is performed outside the computational graph using the `.data` attribute. This is essential for preventing the optimization step itself from being differentiated, thereby preserving memory and maintaining a clean graph for the next forward pass.

## 3. "The Laboratory": Pitfalls & Visuals

### Traps: Buggy Code and Failures

| Trap/Buggy Implementation | Failure Mechanism | Specific Fix/Solution |
| :--- | :--- | :--- |
| **Broadcasting without `keepdim=True`** (`P.sum(dim=1)`). | The normalization sum tensor $\mathbf{S}$ has shape $[V]$ instead of $[V, 1]$. Broadcasting silently aligns $\mathbf{S}$ as $[1, V]$ (a row vector). The subsequent division normalizes the **columns**, not the desired rows, leading to mathematically incorrect probability distributions. | Use `P.sum(dim=1, keepdim=True)` to enforce the output shape $[V, 1]$ (a column vector), correctly triggering row-wise normalization during broadcasting. |
| **Zero Probability Assignment** (e.g., bigram 'JQ' count is 0). | If $P(\text{'Q'}|\text{'J'}) = 0$, the loss term $\log(0)$ evaluates to negative infinity, making the Negative Log Likelihood loss infinite ($\infty$). This implies the model is infinitely surprised by a valid observation. | Implement **Model Smoothing** by adding a fake count (e.g., 1) to all entries in the count matrix $N$, or equivalently, apply **L2 Regularization** $\mathcal{L}_{Reg}$ to the weights $\mathbf{W}$ in the neural network framework. |
| **Using `torch.Tensor` (Capital T)** | If used to create input tensors, it defaults to `float32`. If the data should be character indices, this leads to incorrect data types for indexing or one-hot encoding, which expects integers. | Use `torch.tensor` (lowercase t) for correct data type inference (e.g., `torch.int64`) for integer-based inputs. |

### Magic Numbers (Hyperparameters)

| Parameter | Value(s) Observed | Rationale |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 27 | 26 alphabet characters plus one special start/end token ('.') occupying index 0. |
| **Learning Rate ($\eta$)** | $0.1$ to $50$ | Controls step size in the direction of $-\nabla \mathcal{L}$. Must be large enough to achieve convergence quickly (50 worked in this simple case) but small enough to avoid divergence. |
| **L2 Regularization Strength ($\lambda$)** | $0.01$ | A small factor controlling the trade-off between minimizing NLL loss and minimizing weight magnitude (keeping probabilities uniform). Higher values increase smoothing. |

## 4. Literature

The implemented approach demonstrates two ways to achieve the **Maximum Likelihood Estimate (MLE)** for the bigram probabilities:

1.  **Direct MLE (Counting Method):** Explicitly calculate bigram counts $\mathbf{N}$ and normalize to derive the probability distribution $\mathbf{P}$. This is the direct application of MLE for a bigram Markov model.
2.  **Gradient-based MLE (Neural Network Method):** Minimize the **Average Negative Log Likelihood (Cross-Entropy Loss)**, which is mathematically equivalent to maximizing the likelihood of the training data.

**Connection to Generalized Language Modeling Architectures:**

The simple linear layer ($Z = X_{enc} W$) followed by Softmax is the fundamental prediction head for complex models (like GPT or Transformer decoders).

*   **Mechanism:** In this bigram model, the $\mathbf{W}$ matrix is the entire model parameter space, acting as a direct lookup table for log probabilities.
*   **Scaling:** As the model is complexified (e.g., to accept 10 previous characters), the lookup table approach becomes non-scalable (due to $V^{10}$ states). The gradient-based neural network framework is necessary because it allows the model to compress the input context (via embeddings and subsequent layers) into a dense representation that informs the calculation of $\mathbf{Z}$ (logits), which then retains the same Softmax/NLL pipeline.

The overall architecture demonstrates the minimum viable computational graph for any character or word-level language model: **Context $\to$ Logits $\to$ Softmax $\to$ Probability Distribution $\to$ NLL Loss $\to$ Backward Pass.** This machinery remains identical even when the forward context pathway is replaced by sophisticated deep neural networks, such as recurrent neural networks or the Transformer block.