## 1. Theory, Math & Derivations

### Concepts: Rigorous Definitions

| Concept | Rigorous Definition and Context |
| :--- | :--- |
| **Backpropagation (Backprop)** | A method for computing the gradient of a composite function (the loss) with respect to its inputs (the network parameters) by iteratively applying the **Chain Rule** from the output layer backward to the input layer. On the tensor level, this involves calculating the local derivative of each operation and multiplying it by the global derivative (the gradient of the loss wrt the operation's output). |
| **Leaky Abstraction** | A concept applied to PyTorch’s automatic differentiation (autograd). It denotes that relying solely on `loss.backward()` without understanding the underlying gradient mechanics (e.g., tensor shapes, local derivatives, gradient flow) leads to difficulties in debugging non-optimal performance, numerical instability, or subtle implementation bugs. |
| **Cross-Entropy Loss** | A loss function used in classification tasks, defined as the negative logarithm of the predicted probability assigned to the true class label, typically averaged over the batch. It is derived from the negative log-likelihood of the data under the model's predicted probability distribution (Softmax output). |
| **Batch Normalization (BN)** | A technique introduced to stabilize and accelerate training by normalizing the inputs to each layer (specifically, subtracting the mini-batch mean $\mu$ and dividing by the mini-batch standard deviation $\sigma$). It also applies learned scaling ($\gamma$, gain) and shifting ($\beta$, bias) parameters. |
| **Chain Rule (Tensor Calculus)** | The principle governing gradient flow: if $C = f(B)$ and $B = g(A)$, then $\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial B} \odot \frac{\partial B}{\partial A}$. When dealing with vectors and matrices, multiplication is replaced by matrix multiplication or element-wise operations, often requiring transposes or summation based on dimensionality. |
| **Gradient Summation (Fan-out)** | If a single input variable or tensor element is reused multiple times in the forward graph (a fan-out), the gradients flowing back from all downstream uses must be summed at that node to correctly calculate the total contribution to the loss. This principle is the dual of broadcasting/replication in the forward pass. |

### Formulas, Gradients, and Derivations

Let $B$ be the batch size ($N=32$), $T$ the context length ($T=3$), $E$ the embedding size ($E=10$), $H$ the hidden size ($H=64$), and $C$ the vocabulary size ($C=27$).

#### I. Analytic Backward Pass: Cross-Entropy Loss (Log-Prob $\mathbf{L}_{\text{props}}$ to Logits $\mathbf{L}$)

The derivation proceeds from the definition of the loss $\mathcal{L}$ through the $\text{Softmax}$ function.

**Variables and Dimensions:**
*   Logits: $\mathbf{L} \in \mathbb{R}^{B \times C}$
*   Probabilities (Softmax): $\mathbf{P} \in \mathbb{R}^{B \times C}$
*   Labels: $\mathbf{Y}_B \in \mathbb{Z}^{B}$ (indices of correct classes)

The loss contribution for a single example $i$ is $l_i = -\log(P_{i, y_i})$, where $P_{i, y_i}$ is the probability of the correct label $y_i$.

**Gradients ($D\mathcal{L}/D\mathbf{L}$):**
The analytic derivative $\mathbf{D}_{\mathbf{L}} = \frac{\partial \mathcal{L}}{\partial \mathbf{L}}$ is derived by separating the case where the index $j$ equals the true label $y_i$ or not:

$$(\mathbf{D}_{\mathbf{L}})_{i, j} = \begin{cases} \frac{1}{B} (P_{i, j} - 1) & \text{if } j = y_i \\ \frac{1}{B} P_{i, j} & \text{if } j \neq y_i \end{cases}$$

This can be implemented compactly as a vectorized operation:

$$\mathbf{D}_{\mathbf{L}} = \frac{1}{B} \times \left( \text{Softmax}(\mathbf{L}) - \mathbf{M} \right)$$

Where $\mathbf{M} \in \mathbb{R}^{B \times C}$ is the one-hot encoding mask derived from $\mathbf{Y}_B$.

#### II. Backpropagation through Linear Layer 2 ($\mathbf{L} = \mathbf{H} \mathbf{W}_2 + \mathbf{B}_2$)

$\mathbf{D}_{\mathbf{L}} \in \mathbb{R}^{B \times C}$ is the incoming gradient.

1. **Gradient wrt Hidden State ($\mathbf{H}$):**
$$\mathbf{D}_{\mathbf{H}} = \mathbf{D}_{\mathbf{L}} \mathbf{W}_2^T$$
Where $\mathbf{W}_2 \in \mathbb{R}^{H \times C}$ ($64 \times 27$). $\mathbf{D}_{\mathbf{H}} \in \mathbb{R}^{B \times H}$ ($32 \times 64$).
*Dimension Check*: $(B \times C) \times (C \times H) \rightarrow (B \times H)$.

2. **Gradient wrt Weights ($\mathbf{W}_2$):**
$$\mathbf{D}_{\mathbf{W}_2} = \mathbf{H}^T \mathbf{D}_{\mathbf{L}}$$
Where $\mathbf{H} \in \mathbb{R}^{B \times H}$. $\mathbf{D}_{\mathbf{W}_2} \in \mathbb{R}^{H \times C}$ ($64 \times 27$).
*Dimension Check*: $(H \times B) \times (B \times C) \rightarrow (H \times C)$.

3. **Gradient wrt Bias ($\mathbf{B}_2$):**
The bias $\mathbf{B}_2 \in \mathbb{R}^{C}$ is broadcasted vertically across the batch.
$$\mathbf{D}_{\mathbf{B}_2} = \sum_{i=1}^{B} (\mathbf{D}_{\mathbf{L}})_{i, :}$$
$$\mathbf{D}_{\mathbf{B}_2} = \text{torch.sum}(\mathbf{D}_{\mathbf{L}}, \text{dim}=0)$$
Where $\mathbf{D}_{\mathbf{B}_2} \in \mathbb{R}^{C}$ ($27$).

#### III. Backpropagation through $\tanh$ Activation ($\mathbf{H} = \tanh(\mathbf{H}_{\text{preact}})$)

**Variables and Dimensions:** $\mathbf{H}, \mathbf{H}_{\text{preact}} \in \mathbb{R}^{B \times H}$ ($32 \times 64$).

The local derivative of $a = \tanh(z)$ is $\frac{\partial a}{\partial z} = 1 - a^2$.

$$\mathbf{D}_{\mathbf{H}_{\text{preact}}} = (1 - \mathbf{H} \odot \mathbf{H}) \odot \mathbf{D}_{\mathbf{H}}$$
Where $\odot$ denotes element-wise multiplication. $\mathbf{D}_{\mathbf{H}_{\text{preact}}} \in \mathbb{R}^{B \times H}$.

#### IV. Analytic Backward Pass: Batch Normalization ($\mathbf{Y} = \mathbf{X}_{\text{BN}}$)

Let $\mathbf{X} = \mathbf{H}_{\text{pbn}}$ (input), $\mathbf{Y} = \mathbf{H}_{\text{preact}}$ (output after $\gamma, \beta$), $N$ be the batch size. For simplicity, we use the unbiased variance estimator $1/(N-1)$ (Bessel's correction).

We are concerned with calculating $\mathbf{D}_{\mathbf{X}} = \frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ given $\mathbf{D}_{\mathbf{Y}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}}$.

The overall formula for $\mathbf{D}_{\mathbf{X}}$ (derived by tracing paths through $\mu$, $\sigma^2$, and $\hat{X}$):

$$\mathbf{D}_{\mathbf{X}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left[ \mathbf{D}_{\mathbf{Y}} - \frac{1}{N} \left( \sum_{j=1}^{N} \mathbf{D}_{\mathbf{Y}, j} \right) - \frac{\hat{\mathbf{X}}}{N} \left( \sum_{j=1}^{N} \mathbf{D}_{\mathbf{Y}, j} \odot \hat{\mathbf{X}}_j \right) \right]$$

*   $\mathbf{D}_{\mathbf{X}}, \mathbf{D}_{\mathbf{Y}}, \hat{\mathbf{X}} \in \mathbb{R}^{B \times H}$.
*   $\gamma \in \mathbb{R}^{1 \times H}$ (BN gain, broadcasted).
*   $\sigma^2$ (variance) and the inner summation terms are vectors $\mathbb{R}^{1 \times H}$.

### Why: Explanations of Design Choices

1.  **Bias $B_1$ before BN**: Although the BN mean subtraction renders $B_1$ statistically unnecessary in terms of model capacity, it is included purely as an *exercise* to ensure the manual gradient computation for the linear layer ($\mathbf{D}_{B_1}$) is correct.
2.  **Small Random Bias Initialization**: Initializing biases to small non-zero random values (instead of the typical $0$) is an **engineering trick** used to prevent the potential masking of subtle bugs in gradient calculation, which could occur if the derivative simplifies too much when variables are exactly zero.
3.  **Numerical Stability in Softmax**: The forward pass subtracts the maximum logit per row: $\mathbf{L}_{\text{norm}} = \mathbf{L} - \mathbf{L}_{\text{max}}$. This ensures that exponentiation $e^{\mathbf{L}_{\text{norm}}}$ remains within a manageable range (max exponent is $e^0=1$), preventing floating-point overflow which would otherwise destabilize the Softmax calculation.
4.  **$1/(N-1)$ in BN Variance**: The use of Bessel's correction ($\frac{1}{N-1}$ instead of $\frac{1}{N}$) provides an **unbiased estimate** of the variance. This is preferred because mini-batches are considered small samples of the total population, and dividing by $N$ (the biased estimator) tends to systematically underestimate the true population variance.

---

## 2. Code, Tensors & Internals

### Tensor Tracking and Broadcasting Duality

The core engineering principle derived from the exercise is the **Duality of Summation and Broadcasting**.

| Forward Pass Operation | Input Shape | Output Shape | Backward Pass Operation | Incoming Gradient $\mathbf{D}_{\text{out}}$ | Resulting Gradient $\mathbf{D}_{\text{in}}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sum/Reduction** | $\mathbb{R}^{B \times D}$ | $\mathbb{R}^{1 \times D}$ | **Replication/Broadcast** | $\mathbf{D}_{\text{out}} \in \mathbb{R}^{1 \times D}$ | $\mathbf{D}_{\text{in}} = \mathbf{D}_{\text{out}}$ replicated $B$ times. |
| **Broadcast/Replication** | $\mathbb{R}^{1 \times D}$ | $\mathbb{R}^{B \times D}$ | **Sum/Reduction** | $\mathbf{D}_{\text{out}} \in \mathbb{R}^{B \times D}$ | $\mathbf{D}_{\text{in}} = \sum_{i=1}^{B} (\mathbf{D}_{\text{out}})_{i, :}$. |

**Example: Backprop through $\mathbf{P} = \mathbf{C}_{\text{ounts}} \odot \mathbf{C}_{\text{sum}}^{\text{inv}}$**

1.  $\mathbf{C}_{\text{counts}} \in \mathbb{R}^{B \times C}$, $\mathbf{C}_{\text{sum}}^{\text{inv}} \in \mathbb{R}^{B \times 1}$ (or $\mathbb{R}^{1 \times C}$ if normalized by feature).
2.  Forward operation involves vertical replication (broadcasting) of $\mathbf{C}_{\text{sum}}^{\text{inv}}$ across the $C$ dimension for element-wise multiplication.
3.  Backpropagating to $\mathbf{D}_{\mathbf{C}_{\text{sum}}^{\text{inv}}}$ requires summation across the dimension of replication (axis 1 if column vector, or axis 0 if row vector $\mathbb{R}^{B \times 1}$).

$$\mathbf{D}_{\mathbf{C}_{\text{sum}}^{\text{inv}}} = \text{torch.sum}(\mathbf{C}_{\text{counts}} \odot \mathbf{D}_{\mathbf{P}}, \text{dim}=1, \text{keepdims=True})$$

### Nuance: PyTorch Specifics

1.  **Indexing and Accumulation (Embedding Lookup)**: The lookup operation $\mathbf{M} = \mathbf{C}[X_B]$ involves scattering the gradient $\mathbf{D}_{\mathbf{M}}$ back to the embedding table $\mathbf{C}$. Because a single row of $\mathbf{C}$ (e.g., embedding for character 'A') is likely used multiple times in the batch, the gradients must **additively accumulate** at those indices ($\mathbf{D}_{\mathbf{C}}[\text{index}] \boldsymbol{+}= \mathbf{D}_{\mathbf{M}}[\text{position}]$). This is implemented using nested `for` loops in the source, as a fully vectorized PyTorch equivalent might be complex (`index_add` or manual scatter-add required).
2.  **`view` for Contiguity**: In the forward pass, the $32 \times 3 \times 10$ embedding tensor $\mathbf{M}$ is transformed into the $32 \times 30$ hidden state $\mathbf{MCAT}$ using a `view` operation. In the backward pass, the gradient $\mathbf{D}_{\mathbf{MCAT}}$ is converted back to $\mathbf{D}_{\mathbf{M}}$ simply by applying `view(m.shape)`. This works because `view` is a logical reinterpretation of memory without data copying, and the backward pass simply undoes the shape convention using the exact same underlying memory layout.
3.  **Efficiency and `with torch.no_grad()`**: Once the backward pass is implemented manually, the PyTorch autograd engine is no longer needed for gradient calculation. Wrapping the optimization step in `with torch.no_grad()` tells PyTorch that subsequent operations do not need gradient tracking, enabling the framework to run more efficiently during the parameter update phase.

---

## 3. "The Laboratory": Pitfalls & Visuals

### Traps: Gradient Bugs and Debugging

| Bug/Trap Shown | Explanation of Failure | Specific Fix Implemented |
| :--- | :--- | :--- |
| **Masking Gradients with $B=0$** | If biases are zero, certain mathematical paths that lead to complex derivatives simplify, hiding calculation errors (e.g., if a broadcasted term that should sum to zero in one path only simplifies when zero). | Initialize biases with **small random numbers** to ensure robustness checks. |
| **Loss Clipping (Incorrect Intent)** | Clipping the loss value of an outlier example sets the derivative of that example to zero. | **Physical Effect**: The outlier is entirely ignored by the optimizer, preventing the network from learning from highly informative, high-loss samples. The fix requires implementing **gradient clipping** instead, ensuring gradients are bounded without zeroing the data path. |
| **Incomplete Backprop** | Failing to account for variables that branch (fan-out), such as `B_N_diff`. Calculating the gradient for only one branch results in an incorrect (partial) derivative. | Must use `+=` (plus equals) or explicit summation (`+`) to combine gradients from all parallel branches using that variable. |
| **Floating Point Differences** | Even with correct implementation, manual gradients may not be bitwise identical to autograd gradients. This is due to differing internal computational orders causing minor accumulated floating-point errors. | Use `torch.allclose` (approximate equality) and check that the maximum difference is negligible (e.g., $10^{-9}$). |

### Visuals and Interpretation

**Cross-Entropy Gradient Field ($D_{\mathbf{L}}$) Interpretation:**
The gradient on the logits, $\mathbf{D}_{\mathbf{L}}$, provides an intuitive understanding of the learning force applied to the network.

1.  **Magnitude of Force**: The magnitude of the components of $\mathbf{D}_{\mathbf{L}}$ is proportional to how wrong the network’s prediction was. A perfectly predicted example has $\mathbf{D}_{\mathbf{L}} \approx \mathbf{0}$.
2.  **Push and Pull**: The gradient acts as a set of mechanical forces:
    *   **Pull**: The negative element at the correct index ($P_{i, y_i} - 1$) acts as a strong pulling force, instructing the network to increase the probability (logit) of the correct character.
    *   **Push**: The positive elements (probabilities $P_{i, j}$ where $j \neq y_i$) act as pushing forces, instructing the network to decrease the probabilities (logits) of incorrect characters.
3.  **Balance**: Since $\sum_{j} (\mathbf{D}_{\mathbf{L}})_{i, j} = 0$, the total repulsive force exactly equals the attractive force for any given example.

### Magic Numbers and Context

| Hyperparameter | Value | Context/Derivation |
| :--- | :--- | :--- |
| **Learning Rate (Implicit)** | Used during `p.data += -learning_rate * grad.data`. | Not explicitly defined in the provided source excerpts, but assumed to be tuned for training stability. |
| **BN $\epsilon$** | $10^{-5}$ | Standard numerical stabilizer, preventing division by zero during variance calculation. |
| **Normalization Factor** | $1/N$ or $1/(N-1)$ | Used for loss averaging ($1/N$) and variance calculation ($1/(N-1)$ for unbiased estimate). |

---

## 4. Literature

The implementation connects deeply to standard literature through its optimization of common deep learning components:

*   **Batch Normalization (Ioffe & Szegedy, 2015)**: The derivation highlights the highly non-trivial analytic backward pass. A key simplification derived is that the gradient of the variance $\sigma^2$ with respect to the mean $\mu$ vanishes ($\frac{\partial \sigma^2}{\partial \mu} = 0$) when $\mu$ is defined as the empirical average, simplifying the subsequent calculation of $D\mathcal{L}/D\mathbf{X}$. The choice to use **Bessel's Correction** ($1/(N-1)$) contradicts the original paper's use of $1/N$ during training, mitigating a "train/test mismatch" considered a bug by the implementation author.

*   **Early Language Models (e.g., Bengio et al., 2003)**: The overall architecture—using character embeddings ($\mathbf{C}$) projected into a hidden space ($\mathbf{H}$) and culminating in a Softmax output for prediction—reflects the architecture of foundational neural probabilistic language models.

*   **Matrix Calculus (General)**: The derivation of the matrix multiplication backward pass (for $\mathbf{D}_{\mathbf{W}_2}$ and $\mathbf{D}_{\mathbf{H}}$) formalizes results often found in matrix calculus texts, specifically showing:
    *   $\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \frac{\partial \mathcal{L}}{\partial \mathbf{D}} \mathbf{B}^T$
    *   $\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \mathbf{A}^T \frac{\partial \mathcal{L}}{\partial \mathbf{D}}$.
    The key insight is that these matrix multiplications can be derived using simple scalar chain rule analysis on small examples and verified by **dimensional matching** (ensuring the resulting gradient tensor has the correct shape).

This entire exercise serves as a validation that the complex workings of modern auto-differentiation frameworks, like PyTorch, can be reduced to traceable, mathematically rigorous components, reinforcing the idea that understanding the gradient flow is crucial for becoming a "Backprop Ninja".

***

**Analogy for Gradient Flow (Cross-Entropy):**
Imagine the neural network as a massive, complex **pulley system**. The loss value represents a weight suspended at the very end. The logits are levers at the top of the system. The cross-entropy backward pass applies precisely calibrated **tension** (the gradient $\mathbf{D}_{\mathbf{L}}$) to these levers. At the position corresponding to the correct answer, a specific upward tug is applied, while the incorrect answers receive an opposing downward push. Since the total forces applied sum to zero, the system remains in **equilibrium**, ensuring that every parameter update is efficient and precisely counteracts the current prediction error without unnecessarily shifting the overall probability mass.