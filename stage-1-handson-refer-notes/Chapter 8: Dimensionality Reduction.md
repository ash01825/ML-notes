# Chapter 8: Dimensionality Reduction

Dimensionality Reduction is the process of reducing the number of features (dimensions) in a training dataset. This is crucial when dealing with problems involving thousands or even millions of features per training instance.

## I. Motivation and the Curse of Dimensionality

The **Curse of Dimensionality** refers to the phenomena where high-dimensional spaces are so sparse that the training data is likely to be very far away from most test instances, making predictions unreliable and training slow.

### Goals (Why Use Dimensionality Reduction?)
1.  **Speed:** To speed up a subsequent training algorithm.
2.  **Visualization:** To reduce dimensions to 2D or 3D, allowing visualization of the dataset to gain insights and visually detect patterns like clusters. This is also essential for communicating conclusions to non-data scientists.
3.  **Compression:** To save space and memory.

### Drawbacks
Dimensionality reduction techniques involve tradeoffs:
*   Some information is **lost**, which may degrade the performance of subsequent training algorithms.
*   It can be **computationally intensive**.
*   It adds **complexity** to the Machine Learning pipeline.
*   Transformed features are often **hard to interpret**.

## II. Approaches to Dimensionality Reduction

There are two main families of techniques:

### 1. Projection
This approach projects the training data onto a lower-dimensional subspace.

*   **Example:** Linear Projection (like standard PCA).

### 2. Manifold Learning
Many dimensionality reduction algorithms work by modeling the data as lying on a low-dimensional sub-manifold embedded within the high-dimensional space.

*   This works because in real-world problems, high-dimensional datasets often concentrate around a much lower-dimensional manifold (the **Manifold Hypothesis**).
*   **Example:** Locally Linear Embedding (LLE).

---

## III. Principal Component Analysis (PCA)

PCA is the most popular dimensionality reduction algorithm.

### A. Core Concept: Preserving Variance
1.  **Objective:** PCA first identifies the hyperplane that lies closest to the data.
2.  **Selection:** It chooses the axis that preserves the **maximum amount of variance**.
3.  **Equivalence:** This selection is equivalent to choosing the axis that minimizes the mean squared distance between the original dataset and its projection onto that axis.

### B. Principal Components (PC)
*   The first **Principal Component** ($PC_1$) is the axis that accounts for the largest amount of variance in the training set.
*   The second $PC$ is the axis (orthogonal to the first) that accounts for the largest amount of remaining variance, and so on.
*   There are as many PCs as the number of dimensions in the dataset.

### C. Finding the Principal Components
PCA uses **Singular Value Decomposition (SVD)** to decompose the training set matrix $X$ into the matrix multiplication of three matrices ($U$, $\Sigma$, $V^T$).

$$
X = U \Sigma V^T
$$

*   The matrix $V$ contains the unit vectors that define all the principal components.

### D. Selecting the Dimensionality
The choice of target dimensionality ($d$) is a hyperparameter.

1.  **Visualization:** If the goal is visualization, $d$ is typically set to 2 or 3.
2.  **Explained Variance Ratio:** For compression, the most common approach is to choose the smallest $d$ that preserves a large fraction of the variance (e.g., 95%). The **explained variance ratio** is the ratio of variance preserved along each principal component.

### E. PCA Variants
1.  **PCA for Compression:** Once the data is projected onto $d$ dimensions, the data size is reduced. To restore the original data (decompression), you reverse the projection, though some information is lost.
2.  **Randomized PCA:** A stochastic algorithm that quickly finds an approximation of the first $d$ principal components. It is much faster than standard PCA when the number of dimensions ($n$) or the size of the dataset ($m$) is large.
3.  **Incremental PCA (IPCA):** Allows PCA to be applied incrementally on mini-batches. This is useful for large datasets that do not fit in memory (**out-of-core learning**).

---

## IV. Kernel PCA

Kernel PCA (kPCA) applies the kernel trick (Chapter 5) to perform complex non-linear dimensionality reduction.

*   **Mechanism:** It maps the data implicitly into a high-dimensional feature space using a kernel function (e.g., polynomial, Gaussian RBF) and then performs PCA within that space.
*   **Kernels:** Commonly used kernels include Linear, Polynomial, Gaussian RBF, and Sigmoid.
*   **Hyperparameter Tuning:** Like SVMs, kPCA performance relies on selecting the right kernel and tuning its hyperparameters (e.g., $\gamma$ for RBF or $d$ for polynomial).
*   **Pre-image Problem:** Transforming the reduced dataset back to the original (or close to the original) high-dimensional space is called the pre-image problem. This process is not as simple as reversing the projection in standard PCA.

---

## V. Locally Linear Embedding (LLE)

LLE is a powerful non-linear dimensionality reduction technique and a type of Manifold Learning. It is an unsupervised learning algorithm.

*   **Goal:** LLE maps the training instances to a lower-dimensional space ($d < n$) while attempting to preserve the local relationships between instances.

The algorithm proceeds in two main steps:

### Step 1: Linearly Modeling Local Relationships
For every instance $x^{(i)}$, LLE identifies its $k$ nearest neighbors and attempts to reconstruct $x^{(i)}$ as a linear combination of these neighbors. The optimal reconstruction weights are stored in the weight matrix $W$.

### Step 2: Mapping to Lower Dimension
LLE maps the instances into the $d$-dimensional space (creating $Z$) by keeping the previously computed weights $W$ fixed. It searches for the optimal positions $z^{(i)}$ such that the local relationships encoded by $W$ are preserved as much as possible.

---

## VI. Mind Map Outline

Below is a structured outline designed to serve as a mind map for quick reference:

**Dimensionality Reduction (Chapter 8)**
*   **I. Motivation**
    *   Curse of Dimensionality (Sparsity)
    *   Goals
        *   Speed up Training
        *   Visualization (2D/3D)
        *   Compression (Save Space)
    *   Drawbacks
        *   Information Loss
        *   Increased Complexity
        *   Harder Interpretation
*   **II. Approaches**
    *   Projection
    *   Manifold Learning
        *   Manifold Hypothesis
*   **III. Principal Component Analysis (PCA)**
    *   Goal: Maximize Preserved Variance / Minimize Mean Squared Distance
    *   Principal Components (PCs): Orthogonal Axes of Max Variance
    *   Algorithm: Singular Value Decomposition (SVD)
    *   Selecting Dimension ($d$)
        *   Explained Variance Ratio (e.g., 95%)
    *   Variants
        *   Randomized PCA (Large $n$ or $m$)
        *   Incremental PCA (IPCA, Out-of-Core)
*   **IV. Kernel PCA (kPCA)**
    *   Mechanism: Kernel Trick for Non-Linear Projection
    *   Kernel Choices (RBF, Polynomial, Sigmoid, Linear)
    *   Challenges: Pre-Image Problem
*   **V. Locally Linear Embedding (LLE)**
    *   Type: Manifold Learning (Unsupervised)
    *   Step 1: Local Modeling (Compute weights $W$ from neighbors)
    *   Step 2: Optimal Mapping (Find low-dim $Z$ to preserve $W$)

***