# Top 1% Mastery Guide: Character-Level MLP Language Model

## 1. Theory, Math & Derivations

### Concepts

| Term | Rigorous Definition (as implied by context) | Source |
| :--- | :--- | :--- |
| **Multi-Layer Perceptron (MLP)** | A feedforward neural network structure characterized by an input layer, one or more hidden layers employing a non-linearity (e.g., $\tanh$), and a final output layer. In this context, it takes concatenated embedding vectors as input and outputs a probability distribution (logits) over the next character. | |
| **Embedding** | A mapping of a discrete, high-dimensional token index (character index) to a continuous, dense, low-dimensional feature vector $\mathbf{c} \in \mathbb{R}^D$. These vectors, stored in the matrix $\mathbf{C}$, are optimized via backpropagation to position related tokens close together in the feature space, enabling knowledge transfer and generalization. | |
| **Block Size ($T$)** | A hyperparameter defining the fixed length of the previous context (sequence of characters) used as input to predict the subsequent character. Increasing $T$ exponentially increases context possibilities in sparse models (e.g., bigram), a problem the MLP addresses. | |
| **Backpropagation** | The algorithmic application of the chain rule to efficiently calculate the gradient of the scalar loss function ($\mathcal{L}$) with respect to all parameters (weights $\mathbf{W}$, biases $\mathbf{B}$, embedding matrix $\mathbf{C}$). This process populates the `.grad` attributes necessary for parameter updates via gradient descent. | |
| **Negative Log Likelihood (NLL) Loss** | The loss function minimized during training, equivalent to maximizing the log likelihood of the training data. For a correct target $y$, the NLL loss $\mathcal{L} = -\log(\mathbf{P}_{y})$, penalizing low probability assignments ($\mathbf{P}_{y}$) to the true character. | |
| **Cross-Entropy Loss ($\mathcal{L}$)** | The standard loss function for multi-class classification, numerically equivalent to NLL when using one-hot targets. PyTorch's `F.cross_entropy` efficiently combines the softmax operation and NLL calculation, offering crucial numerical stability by implementing the max-shifting trick. | |

### Formulas, Dimensions, and Forward Pass Derivation (Standard Variables)

We define the following tensor dimensions:
*   $B$: Mini-batch size (e.g., 32).
*   $T$: Block size/Context length (e.g., 3).
*   $V$: Vocabulary size (e.g., 27).
*   $D$: Embedding dimensionality (e.g., 10).
*   $H$: Hidden layer size (e.g., 200).
*   $D_{in}$: Input dimensionality to the hidden layer, $D_{in} = T \cdot D$.

1.  **Input Indices and Embedding Matrix:**
    $$\mathbf{X} \in \mathbb{Z}^{B \times T}$$
    $$\mathbf{C} \in \mathbb{R}^{V \times D}$$

2.  **Embedding Lookup:** (Equivalent to one-hot encoding $\mathbf{X}$ and multiplying by $\mathbf{C}$):
    $$\mathbf{E} = \mathbf{C}[\mathbf{X}]$$
    $$\mathbf{E} \in \mathbb{R}^{B \times T \times D}$$

3.  **Input Flattening/Concatenation (View Operation):**
    $$\mathbf{E}_{\text{flat}} = \text{View}(\mathbf{E}, [B, D_{in}])$$
    $$\mathbf{E}_{\text{flat}} \in \mathbb{R}^{B \times D_{in}}$$

4.  **Hidden Layer Pre-Activation (Affine Transformation):**
    The weights $\mathbf{W}_1$ and biases $\mathbf{B}_1$ are the learnable parameters for the hidden layer.
    $$\mathbf{L}_1 = \mathbf{E}_{\text{flat}} \mathbf{W}_1 + \mathbf{B}_1$$
    Where:
    *   $\mathbf{W}_1 \in \mathbb{R}^{D_{in} \times H}$ (e.g., $30 \times 200$).
    *   $\mathbf{B}_1 \in \mathbb{R}^{1 \times H}$ (Bias is broadcasted across the batch dimension).
    *   $\mathbf{L}_1 \in \mathbb{R}^{B \times H}$

5.  **Hidden Layer Activation:**
    The $\tanh$ non-linearity is applied element-wise.
    $$\mathbf{H} = \tanh(\mathbf{L}_1)$$
    $$\mathbf{H} \in \mathbb{R}^{B \times H}$$

6.  **Output Logits:**
    The weights $\mathbf{W}_2$ and biases $\mathbf{B}_2$ are the learnable parameters for the output layer.
    $$\mathbf{L} = \mathbf{H} \mathbf{W}_2 + \mathbf{B}_2$$
    Where:
    *   $\mathbf{W}_2 \in \mathbb{R}^{H \times V}$ (e.g., $200 \times 27$).
    *   $\mathbf{B}_2 \in \mathbb{R}^{1 \times V}$ (Bias is broadcasted).
    *   $\mathbf{L} \in \mathbb{R}^{B \times V}$

7.  **Loss Function (Optimized Implementation):**
    The loss $\mathcal{L}$ is calculated directly from the logits and the target indices $\mathbf{Y} \in \mathbb{Z}^B$.
    $$\mathcal{L} = F.\text{cross\_entropy}(\mathbf{L}, \mathbf{Y})$$

### Why: Design Choices

| Design Choice | Rationale/Engineering Insight | Source |
| :--- | :--- | :--- |
| **Hidden Layer $\tanh$** | $\tanh$ is a standard non-linearity used in early neural networks. It squashes activation values into the range $[-1, 1]$. | |
| **Embedding Size $D=2$ (Initial)** | Chosen initially purely for the ability to **visualize** the learned structure of the character embeddings in a 2D scatter plot. | |
| **Increasing $D$ to $10$** | Increased because the model showed signs of **underfitting** (training loss $\approx$ validation loss), suggesting that the low dimensionality of $D=2$ was a **bottleneck**, preventing the network from adequately separating character features. | |
| **Exponential Learning Rate Search** | Using `torch.linspace` across the *exponent* ($\log_{10}(\text{LR})$) ensures that the search space is **exponentially spaced** (e.g., $0.001, 0.01, 0.1, 1.0$). This is crucial because optimization behavior changes drastically across orders of magnitude of LR. | |

### Gradients and Backpropagation Nuance

The derivative calculus details are bypassed in favor of specialized, fused implementations:

1.  **$\tanh$ Backward Pass:** While the forward pass involves complicated mathematical expressions (e.g., combining exponentials), the derivative simplifies analytically to $\frac{d(\tanh(x))}{dx} = 1 - \tanh^2(x)$. PyTorch reuses the calculated $\mathbf{H}$ (the forward result) in the backward pass to efficiently compute the local gradient, avoiding re-calculation of complex terms.
2.  **Cross-Entropy Backward Pass:** Using `F.cross_entropy` clusters the Softmax and NLL calculation. This results in a simpler derivative expression and allows the use of **fused kernels**. Fused kernels combine multiple operations (like exponentiation, summing, dividing, logging) into a single, optimized operation on the GPU, greatly enhancing backward pass efficiency.

## 2. Code, Tensors & Internals

### Tensor Tracking and Memory Operations

The core engineering trick in the forward pass involves manipulating tensor views to avoid expensive memory copies.

| Code Block | Operation | Initial Tensor Shape | Output Shape | Tensor Tracking Nuance | Performance Implication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ix = torch.randint(0, X.shape, (B,))` | Mini-batch index sampling. | $X_{\text{train}}: [228k, 3]$ | $ix:$ | Integer Long tensor used for indexing. | Instantaneous indexing setup. |
| `emb = C[X[ix]]` | Embedding lookup. | $X_{\text{train}}[ix]:$ $C: [27, D]$ | $emb: [32, 3, D]$ | Indexing plucks out $32 \times 3$ embedding vectors. | Highly efficient lookup mechanism. |
| **`emb.view(-1, D_in)`** | **Flattening/Concatenation**. | $emb: [32, 3, D]$ | $h_{\text{pre}}: [32, D_{in}]$ | **Critical view operation.** $\mathbf{-1}$ tells PyTorch to infer the batch dimension (32). The `view` operation manipulates *strides* and *shape*. | **Zero-cost, extremely efficient.** No new memory storage is allocated or copied. The resulting tensor logically reinterprets the original memory layout. |
| `h = torch.tanh(...) + B1` | Hidden layer calculation. | $\mathbf{L}_1: [32, H]$ $\mathbf{B}_1: [H]$ | $\mathbf{H}: [32, H]$ | **Broadcasting:** $\mathbf{B}_1$ (1D vector) aligns on the right, effectively becoming $1 \times H$, and is copied 32 times vertically, adding the identical bias to every row in the batch. | Standard optimized element-wise and matrix operations. |

### PyTorch Specifics and Performance

1.  **`view` vs. `torch.cat` (Storage and Strides):**
    When transforming an embedding tensor $\mathbf{E} \in \mathbb{R}^{B \times T \times D}$ into $\mathbf{E}_{\text{flat}} \in \mathbb{R}^{B \times D_{in}}$:
    *   `E.view(-1, D_in)` is preferred because PyTorch tensors are stored in computer memory as a 1D vector (**storage**). `view` simply changes how this 1D storage is interpreted (the **shape** and **strides** attributes) without moving or copying the physical data.
    *   `torch.cat` (e.g., using `torch.unbind` first) is inefficient because it requires creating a **new underlying storage** (new memory allocation) to physically concatenate the separate embedding chunks, despite achieving the same numerical result.

2.  **`register_buffer` (Not explicitly used, but implied context):** Although not shown, parameters like $C, W_1, B_1, W_2, B_2$ are defined as parameters requiring gradients. A tensor that is part of the model state but **does not** require a gradient (e.g., a constant scaling factor or the indices of the vocabulary) would typically be registered using `register_buffer` in a class-based definition, ensuring it is saved and loaded with the model, but skipped during backpropagation.

3.  **Softmax Implementation (`F.cross_entropy` vs. Manual):**
    PyTorch's `F.cross_entropy` is crucial for avoiding three sources of inefficiency/error encountered in a manual implementation:
    *   **Memory Overhead:** It avoids creating three large intermediate tensors (`counts`, `sum`, `probs`) in memory.
    *   **Numerical Overflow:** It prevents $e^L$ from overflowing to $\text{inf}$ when $L$ is large by performing max-shifting internally.
    *   **Backward Pass Efficiency:** It utilizes simplified analytical derivatives and fused kernel optimizations.

## 3. "The Laboratory": Pitfalls & Visuals

### Traps (Physical Failure Mechanisms)

| Trap/Bug | Failure Mechanism | Specific Fix/Remedy |
| :--- | :--- | :--- |
| **Positive Logit Overflow** | Logits ($L$) become large positive numbers (e.g., 100). Exponentiation ($\exp(L)$) causes floating-point overflow, resulting in $\text{inf}$ counts. Normalizing $\frac{\text{inf}}{\sum \text{inf}}$ yields $\text{NaN}$ probabilities. | **Max-Shifting:** Internally, `F.cross_entropy` calculates $L_{\text{shifted}} = L - \max(L)$ before exponentiation. This ensures the maximum exponential argument is $0$, maintaining numerical stability without altering the resulting probabilities. |
| **Integer/Float Multiplication** | Attempting $\text{Long Tensor} \cdot \text{Float Tensor}$. | PyTorch requires explicit casting of integer tensors to floating-point (`.float()`) for mathematical operations with float parameters. |
| **Underfitting** | When training loss is roughly equal to validation/development loss. The model capacity is too low relative to the complexity of the data. | **Increase Model Capacity:** Increase the hidden layer size ($H$) (e.g., 100 to 300) or increase the embedding dimensionality ($D$) (e.g., 2 to 10), as $D$ was identified as a potential bottleneck. |

### Visuals: Chart Reading

1.  **Log Loss Plot over Steps:** When plotting $\log_{10}(\mathcal{L})$ versus step count $i$, the noise (thickness) is inherent due to **Stochastic Gradient Descent (SGD) with mini-batches**. Each mini-batch gradient is an estimate of the full gradient, causing the optimization path to be jagged rather than smooth. Using the log scale ($\log_{10}$) effectively squashes the large initial drops, making the slower, late-stage convergence visible and easier to track.

2.  **Learning Rate Exponent vs. Loss Plot (LR Finder):**
    This chart is read to find the **steepest descent region** before instability. The "valley" where loss decreases rapidly but remains stable (e.g., $10^{-1}$) is the optimal LR range. Choosing an LR slightly before the loss curve begins to flatten (too slow) or explode (too fast) ensures fast convergence. The explosion point confirms that the current optimization step is too large, potentially causing instability and divergence.

### Magic Numbers (Hyperparameters)

The primary set of hyperparameters used in the highly optimized phase included:
*   Embedding Dimensionality ($D$): 10.
*   Hidden Layer Size ($H$): 200 neurons.
*   Context/Block Size ($T$): 3 characters.
*   Mini-Batch Size ($B$): 32 examples.
*   Learning Rate Schedule: Initial $\text{LR}=0.1$ for $100,000$ steps, followed by a **learning rate decay** to $\text{LR}=0.01$ for the subsequent $100,000$ steps. The decay is necessary to settle the optimization into a finer minimum once the initial rapid descent plateaus.

## 4. Literature

The implemented model directly adopts the architecture and training philosophy outlined in the seminal paper **Bengio, et al. (2003)**:

1.  **Distributed Feature Vectors (Embeddings):** Bengio et al. proposed associating each word with a low-dimensional feature vector. The Makemore model implements this concept by creating the embedding matrix $\mathbf{C}$. The use of dense embeddings facilitates **knowledge transfer** across similar tokens (e.g., A, E, I, O, U are clustered), allowing generalization to phrases unseen during training.
2.  **Architecture:** The structure—embedding lookup $\rightarrow$ concatenated input $\rightarrow$ hidden $\tanh$ layer $\rightarrow$ output softmax/logits—is the defining feature of the Neural Network Language Model (NNLM) proposed in the paper. The source confirms that the high computational cost in the original paper, located in the final layer (due to 17,000 words), corresponds in the character model to the $H \times V$ weight matrix $\mathbf{W}_2$.
3.  **Objective Function:** Both the original NNLM and the Makemore implementation rely on maximizing the log likelihood of the training data, achieved by minimizing the Negative Log Likelihood loss (Cross-Entropy) via backpropagation.

The architectural departure is only in scope: Bengio 2003 was a word-level model (17,000 tokens), whereas Makemore is a character-level model (27 tokens).

***

**Analogy for Numerical Stability (Max-Shifting):**

Calculating the Softmax $\mathbf{P} = \exp(\mathbf{L}) / \sum \exp(\mathbf{L})$ is analogous to adjusting a highly sensitive scale. If the inputs (logits $L$) are too heavy (large positive), the scale overflows and breaks ($\text{NaN}$ error). The **max-shifting trick** (subtracting $\max(L)$ from all $L$) is like resetting the zero point of the scale so that the heaviest input is now zero. This ensures that the scale never overflows, yet because we are only shifting all inputs by a constant amount, the ratio (which determines the final probability $\mathbf{P}$) remains mathematically identical.