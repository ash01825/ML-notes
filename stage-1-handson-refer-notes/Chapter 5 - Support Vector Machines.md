## Comprehensive Notes: Chapter 5 - Support Vector Machines

A Support Vector Machine (SVM) is a powerful, versatile Machine Learning model that can perform linear or nonlinear classification, regression, and even outlier detection. SVMs are particularly well-suited for classifying complex small- or medium-sized datasets.

### I. Linear SVM Classification

The foundational idea of linear SVMs is to fit the **widest possible "street"** (or margin) between the classes, known as large margin classification.

#### A. Key Concepts

*   **Large Margin Classification:** The decision boundary separates two classes and stays as far away from the closest training instances as possible, fitting the widest street between them.
*   **Support Vectors:** These are the training instances located directly on the edge of the "street" (the margin). They are the instances that fully determine (or "support") the decision boundary. Adding more training instances "off the street" does not affect the boundary.
*   **Feature Scaling:** SVMs are sensitive to the feature scales. If features are not scaled (e.g., using `StandardScaler`), the resulting "widest possible street" may be skewed, leading to a suboptimal decision boundary.

#### B. Hard Margin vs. Soft Margin Classification

1.  **Hard Margin Classification:**
    *   Requires that all instances must strictly be "off the street" and on the correct side.
    *   This approach only works if the data is perfectly linearly separable.
    *   It is very sensitive to outliers; a single outlier can prevent a perfect separation or lead to a poorly generalized model.

2.  **Soft Margin Classification:**
    *   A more flexible model that seeks a good balance between maximizing the width of the street and limiting **margin violations** (instances that end up in the middle of the street or on the wrong side).
    *   Controlled by the **hyperparameter C**:
        *   **Low C:** Results in a wider margin but allows more margin violations (often generalizes better).
        *   **High C:** Results in a narrower margin but fewer margin violations (might overfit the training data).

### II. Nonlinear SVM Classification

When dealing with non-linearly separable data, SVMs can be adapted using two main techniques.

#### A. The Kernel Trick

The kernel trick is a mathematical technique that allows implicitly mapping instances into a very high-dimensional space, enabling linear decision boundaries in the high-dimensional space that correspond to complex nonlinear decision boundaries in the original space.

*   **Common Kernels**:
    *   **Linear:** $K(\mathbf{a}, \mathbf{b}) = \mathbf{a}^T \mathbf{b}$.
    *   **Polynomial:** $K(\mathbf{a}, \mathbf{b}) = (\gamma \mathbf{a}^T \mathbf{b} + r)^d$. The hyperparameters $d$ (degree) and $r$ control this kernel.
    *   **Gaussian RBF (Radial Basis Function):** $K(\mathbf{a}, \mathbf{b}) = \exp(-\gamma \| \mathbf{a} - \mathbf{b} \|^2)$.
    *   **Sigmoid:** $K(\mathbf{a}, \mathbf{b}) = \tanh(\gamma \mathbf{a}^T \mathbf{b} + r)$.
*   **Kernel Hyperparameters:** The kernel $\gamma$ (gamma) acts like a regularization parameter for the RBF kernel. If $\gamma$ is high, the model is restricted, potentially underfitting. If $\gamma$ is low, the model is less restricted, potentially overfitting.
*   **Mercer's Theorem:** If a function $K(\mathbf{a}, \mathbf{b})$ meets specific mathematical conditions (Mercer’s conditions), there exists a transformation $\phi$ such that $K(\mathbf{a}, \mathbf{b}) = \phi(\mathbf{a})^T \phi(\mathbf{b})$, even if $\phi$ is unknown or maps to an infinite-dimensional space (as in the RBF kernel).

#### B. Similarity Features (Manual Transformation)

Another approach involves manually adding features computed using a **similarity function** (like the Gaussian RBF) to measure how much each instance resembles a specific *landmark*.

*   The simplest method is setting a landmark at the location of every instance in the dataset.
*   This transformation increases dimensionality, increasing the chances the resulting training set will be linearly separable.
*   **Drawback:** If the training set has $m$ instances and $n$ features, the transformed set will have $m$ instances and $m$ features (if original features are dropped), leading to a huge number of features for large datasets.

### III. SVM Under the Hood (Mathematical Foundations)

#### A. Notations

For SVMs, the bias term is called $b$, and the feature weights vector is called $\mathbf{w}$. No bias feature $x_0 = 1$ is added to the input feature vectors.

#### B. Decision Function and Predictions

The linear SVM classifier predicts the class of a new instance $\mathbf{x}$ by computing the decision function: $\mathbf{w}^T \mathbf{x} + b = w_1 x_1 + \cdots + w_n x_n + b$.

*   If the result is **positive** ($> 0$), the predicted class $\hat{y}$ is the positive class (1).
*   If the result is **negative** ($\le 0$), the predicted class $\hat{y}$ is the negative class (0).
*   The dashed lines representing the margin occur where the decision function equals 1 or –1.

#### C. Training Objective (Optimization)

Training a linear SVM classifier means finding the values of $\mathbf{w}$ and $b$ that maximize the margin.

1.  **Maximizing the Margin:** The size of the margin is inversely proportional to the norm of the weight vector, $\|\mathbf{w}\|$. Therefore, we minimize $\|\mathbf{w}\|$ (or equivalently, $\frac{1}{2}\mathbf{w}^T\mathbf{w}$).
2.  **Hard Margin Linear SVM Objective:** This is expressed as a constrained optimization problem:
    $$\text{minimize}_{\mathbf{w}, b} \quad \frac{1}{2} \mathbf{w}^T \mathbf{w}$$
    $$\text{subject to} \quad t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1 \quad \text{for } i = 1, 2, \dots, m$$
    (where $t^{(i)} = -1$ for negative instances and $t^{(i)} = 1$ for positive instances).

3.  **Soft Margin Linear SVM Objective:** To allow for margin violations, a *slack variable* $\zeta^{(i)}$ (zeta) is introduced for every instance:
    $$\text{minimize}_{\mathbf{w}, b, \zeta} \quad \frac{1}{2} \mathbf{w}^T \mathbf{w} + C \sum_{i=1}^m \zeta^{(i)}$$
    $$\text{subject to} \quad t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1 - \zeta^{(i)} \quad \text{and } \zeta^{(i)} \ge 0 \quad \text{for } i = 1, 2, \dots, m$$
    (where $C$ is the hyperparameter).

#### D. The Dual Problem and Kernelization

The hard margin and soft margin problems are **convex Quadratic Programming (QP) problems**. Instead of solving the primal problem directly, it is often preferred to solve the **dual problem**.

*   Both the primal and dual problems share the same solution under the conditions met by the SVM problem.
*   The dual problem for the soft margin objective is:
    $$\text{minimize}_{\mathbf{\alpha}} \quad \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha^{(i)} \alpha^{(j)} t^{(i)} t^{(j)} \mathbf{x}^{(i)T} \mathbf{x}^{(j)} - \sum_{i=1}^m \alpha^{(i)}$$
    $$\text{subject to} \quad 0 \le \alpha^{(i)} \le C \quad \text{for } i = 1, 2, \dots, m \quad \text{and} \quad \sum_{i=1}^m \alpha^{(i)} t^{(i)} = 0$$
*   **Benefit of the Dual Problem:** The input feature vectors only appear in dot products (e.g., $\mathbf{x}^{(i)T} \mathbf{x}^{(j)}$). If the data is transformed using $\phi(\mathbf{x})$, these dot products become $\phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(j)})$, which can be replaced by the kernel function $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ if a kernel is used. This allows the Kernel Trick to be applied.
*   **Support Vectors in the Dual:** Only the instances corresponding to $\alpha^{(i)} > 0$ are the support vectors. Instances "off the street" (correctly classified, outside the margin) have $\alpha^{(i)} = 0$.

### IV. SVM Capabilities Beyond Classification

*   **Regression (SVR):** SVMs can also perform regression by reversing the objective: instead of trying to fit the largest possible street *between* two classes, SVR tries to fit as many instances as possible *on* the street while limiting margin violations. The width of the street is controlled by the hyperparameter $\epsilon$ (epsilon).
*   **Outlier Detection:** SVMs can be used for outlier detection.
*   **Online SVMs:** These learn incrementally by feeding data instances sequentially. `SGDClassifier` can be used to train linear SVMs using Stochastic Gradient Descent (SGD). For linear SVMs, the `LinearSVC` class is often faster than `SVC`. For large-scale nonlinear problems, neural networks may be preferred.

---
### Mind Map Structure: Support Vector Machines (Chapter 5)

This outline serves as a mind map, detailing the hierarchy of concepts discussed in the chapter.

**CHAPTER 5: SUPPORT VECTOR MACHINES (SVM)**
***Core Capabilities:*** Classification, Regression, Outlier Detection

**I. LINEAR SVM CLASSIFICATION**
*   **A. Core Idea:** Large Margin Classification (Fitting the widest street)
    *   **Support Vectors:** Instances defining the margin
*   **B. Requirements:** Feature Scaling is crucial
*   **C. Margin Types**
    *   **Hard Margin:** No margin violations allowed (Sensitive to outliers; only for linearly separable data)
    *   **Soft Margin:** Balance margin width vs. violations (More flexible)
        *   ***Hyperparameter C:*** Controls tolerance for violations (Low C = wide margin, more regularization; High C = narrow margin, less regularization)

**II. NONLINEAR SVM CLASSIFICATION**
*   **A. The Kernel Trick**
    *   Implicit mapping to a higher-dimensional feature space
    *   **Common Kernels:** Linear, Polynomial, Gaussian RBF, Sigmoid
    *   ***Hyperparameter $\gamma$:*** Controls influence of individual training examples (acts as a regularizer)
    *   **Mercer's Theorem:** Guarantees existence of mapping function $\phi$
*   **B. Similarity Features**
    *   Manually adding features (e.g., Gaussian RBF) based on distance to *landmarks*
    *   Simplest method: Use every instance as a landmark

**III. SVM UNDER THE HOOD (MATHEMATICS)**
*   **A. Decision Function:** $\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$
*   **B. Hard Margin Objective (Primal Problem):**
    *   Minimize $\frac{1}{2} \mathbf{w}^T \mathbf{w}$
    *   Subject to: $t^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \ge 1$
*   **C. Soft Margin Objective (Primal Problem):**
    *   Minimize $\frac{1}{2} \mathbf{w}^T \mathbf{w} + C \sum \zeta^{(i)}$ (QP Problem)
*   **D. Dual Problem:** Preferred formulation for training (allows Kernel Trick)
    *   Solution determined by support vectors ($\alpha^{(i)} > 0$)

**IV. SVM VARIANTS & PRACTICE**
*   **Online SVMs:** Use incremental learning (e.g., Stochastic Gradient Descent using `SGDClassifier` or Pegasos).
*   **Regression (SVR):** Uses $\epsilon$ hyperparameter to define an insensitive region (the street).