## Chapter 4: Training Models

This chapter focuses on understanding the internal mechanics of models and their training algorithms, moving beyond treating them as "black boxes". Understanding these mechanics is essential for efficient debugging, error analysis, and selection of models and hyperparameters, particularly for grasping the concepts used in Neural Networks (Part II).

To fully understand the mathematical content, a reasonable understanding of college-level mathematics, including calculus (partial derivatives) and linear algebra (vectors, matrices, transposition, multiplication, and inversion), is assumed.

### I. Linear Regression

A Linear Regression model makes a prediction by computing a weighted sum of the input features, plus a constant term called the bias term (or intercept term).

#### 1. Model Prediction and Cost Function

| Concept | Description |
| :--- | :--- |
| **Model Prediction** | $\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$ |
| $n$ | The number of features. |
| $x_i$ | The $i^{th}$ feature value. |
| $\theta_j$ | The $j^{th}$ model parameter, including the bias term $\theta_0$. |
| **Vectorized Form** | The prediction is typically represented as $\mathbf{\hat{y}} = \mathbf{X}\mathbf{\theta}$. |
| **Cost Function** | The Mean Squared Error ($\text{MSE}$) is typically used for optimization. |

The MSE for a Linear Regression hypothesis $h_{\mathbf{\theta}}$ on a training set $\mathbf{X}$ is calculated as:
$$\text{MSE}(\mathbf{X}, h_{\mathbf{\theta}}) = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{\theta}^{\intercal}\mathbf{x}^{(i)} - y^{(i)})^2 \text{}$$

#### 2. Training Approaches

The two primary ways to train a Linear Regression model are using a direct **closed-form equation** or an **iterative optimization approach** like Gradient Descent (GD).

| Approach | Method | Description |
| :--- | :--- | :--- |
| **Direct Solution** | **The Normal Equation** | Directly computes the model parameters ($\mathbf{\hat{\theta}}$) that minimize the cost function. |
| **Iterative Optimization** | **Gradient Descent (GD)** | Gradually tweaks the model parameters to minimize the cost function over the training set. |

---

#### The Normal Equation

The Normal Equation provides the closed-form solution for the optimal parameters $\mathbf{\theta}$:
$$\mathbf{\hat{\theta}} = (\mathbf{X}^{\intercal}\mathbf{X})^{-1}\mathbf{X}^{\intercal}\mathbf{y} \text{}$$

| Characteristic | Detail |
| :--- | :--- |
| **Computational Complexity** | Inverting the $\mathbf{X}^{\intercal}\mathbf{X}$ matrix is typically $O(n^{2.4})$ to $O(n^3)$ (if $n$ is the number of features). |
| **Scaling** | Scales poorly with the number of features ($n$). It is fast when the number of training instances ($m$) is small or when $n$ is small. |
| **Data Handling** | Requires loading the entire dataset into memory to compute $\mathbf{X}^{\intercal}\mathbf{X}$, which is infeasible for huge datasets. |

---

### III. Gradient Descent (GD)

Gradient Descent is an iterative optimization process that finds the optimal parameters by moving in the direction of the steepest descent, defined by the negative gradient of the cost function.

#### 1. General Principles

*   **Learning Rate ($\eta$):** The critical hyperparameter that determines the size of the steps.
    *   **Too Small:** Requires many iterations to converge, taking too long.
    *   **Too High:** May overshoot the minimum, jump across the valley, or diverge entirely.
*   **Cost Function Shape:** For Linear Regression's MSE, the cost function is a convex parabola (like a bowl), guaranteeing that GD will converge to the global minimum (assuming a good learning rate).
*   **Gradient Vector:** Contains all the partial derivatives of the cost function, computed in one step using the formula:
    $$\nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta}) = \frac{2}{m} \mathbf{X}^{\intercal} (\mathbf{X}\mathbf{\theta} - \mathbf{y}) \text{}$$
*   **Step Size:** The rule for updating parameters $\mathbf{\theta}$ is: $\mathbf{\theta}^{(\text{next step})} = \mathbf{\theta} - \eta \nabla_{\mathbf{\theta}} \text{MSE}(\mathbf{\theta})$.

#### 2. Gradient Descent Variants

| Variant | Gradient Computation | Speed / Convergence Path | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Batch GD (BGD)** | Uses the **entire** training set ($\mathbf{X}$) at every step. | Slow for large $m$. Path stops exactly at the global minimum. | Guaranteed convergence for convex functions, but inefficient for large $m$. |
| **Stochastic GD (SGD)** | Uses a **single, randomly selected instance** at every step. | Fast. Path is erratic, jumping around the minimum. | Can escape local minima (due to randomness). Requires a **learning schedule** to gradually reduce the learning rate to settle near the minimum. |
| **Mini-batch GD (MBGD)** | Uses small, random subsets (**mini-batches**). | Often faster than BGD and SGD due to hardware optimizations (GPUs). Path is less erratic than SGD, ending up closer to the minimum. | Better performance than SGD, but may struggle more than SGD to escape local minima. |

### IV. Polynomial Regression

Polynomial Regression is used when data is too complex for a simple linear model. It adds powers of existing features as new features (e.g., $x_1^2$, $x_1^3$, etc.) and then trains a Linear Model on the extended feature set.

*   **Trade-off:** High-degree Polynomial Regression models can fit training data closely but often severely **overfit** it.
*   **Underfitting:** Occurs when the model is too simple to capture the underlying patterns in the data.

#### Learning Curves

Learning curves plot the model's performance on the training set and the validation set as a function of the training set size.

| Problem | Training Error | Validation Error | Learning Curve Appearance | How to Fix |
| :--- | :--- | :--- | :--- | :--- |
| **Underfitting** | High | High (close to training error) | Both curves reach a high plateau and are close together. | Use a more complex model or better features; adding more training data will **not** help. |
| **Overfitting** | Low | High (large gap from training error) | Training error decreases significantly below validation error. Large gap between curves. | Use regularization (Ridge, Lasso, Elastic Net) or gather more training data. |

### V. Regularized Linear Models

Regularization constrains the weights of the model to reduce complexity and the risk of overfitting.

| Model | Regularization Type | Cost Function Term Added | Effect |
| :--- | :--- | :--- | :--- |
| **Ridge Regression** | $l_2$ norm (squared weights) | $\alpha \sum_{i=1}^{n} \theta_i^2$ | Pushes weights toward zero but does not eliminate them entirely. Serves as a good default choice. |
| **Lasso Regression** | $l_1$ norm (absolute weights) | $\alpha \sum_{i=1}^{n} |\theta_i|$ | Tends to eliminate weights of the least important features (sets them to zero), resulting in a **sparse model** and automatic feature selection. |
| **Elastic Net** | Mix of $l_1$ and $l_2$ | $\alpha (r \sum_{i=1}^{n} |\theta_i| + \frac{1-r}{2} \sum_{i=1}^{n} \theta_i^2)$ | A combination of Ridge and Lasso. Generally preferred over Lasso when features are highly correlated or when $m > n$. |

### VI. Logistic Regression (Classification)

Logistic Regression is commonly used for binary classification tasks, often framed as predicting a probability.

#### 1. Estimating Probabilities

*   The model calculates a weighted sum of inputs (like Linear Regression) and then passes this result through the **logistic function**.
*   **Logistic Function ($\sigma(\cdot)$):** A sigmoid (S-shaped) function that outputs a number between 0 and 1.
    $$\sigma(z) = \frac{1}{1 + \exp(-z)} \text{}$$
*   **Estimated Probability:** $p = h_{\mathbf{\theta}}(\mathbf{x}) = \sigma(\mathbf{x}^{\intercal}\mathbf{\theta})$.

#### 2. Training and Cost Function

*   **Cost Function:** Uses the **log loss** (or cross entropy). This penalizes the model heavily when it estimates a probability close to 0 for a positive instance, or a probability close to 1 for a negative instance.
*   **Convexity:** The log loss cost function is convex, meaning Gradient Descent (or other optimization algorithms) is **guaranteed** to find the global minimum. There is no known closed-form equation (like the Normal Equation) for this cost function.

#### 3. Decision Boundaries

The decision boundary is the point where the estimated probability equals 50%. If the estimated probability is higher than 50%, the model predicts the positive class (1); otherwise, it predicts the negative class (0).

### VII. Softmax Regression (Multiclass Classification)

Softmax Regression (also called Multinomial Logistic Regression) is a generalization of Logistic Regression that handles multiple classes directly.

#### 1. Estimating Probabilities

*   The model first computes a score $s_k(\mathbf{x})$ for each class $k$. These scores are called **logits**.
*   **Softmax Function:** The scores are run through the softmax function to estimate the probability $p_k$ that instance $\mathbf{x}$ belongs to class $k$. The resulting probabilities are between 0 and 1 and sum up to 1.
    $$p_k = \sigma(\mathbf{s}(\mathbf{x}))_k = \frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K} \exp(s_j(\mathbf{x}))} \text{}$$

#### 2. Training and Cost Function

*   **Cost Function:** The objective is to estimate a high probability for the target class, which is achieved by minimizing the **cross entropy** cost function.
*   **Decision Boundaries:** The decision boundaries between any two classes are linear.

***

## Mind Map: Chapter 4 — Training Models

I can provide a hierarchical textual outline that serves as an effective mind map for navigation and reference:

**CHAPTER 4: TRAINING MODELS**

1.  **LINEAR REGRESSION (Predicting Values)**
    *   **Model:** $\hat{y} = \mathbf{x}^{\intercal}\mathbf{\theta}$ (Weighted sum + bias)
    *   **Cost Function:** Mean Squared Error (MSE)
    *   **Training Methods**
        *   **A. The Normal Equation (Direct)**
            *   Formula: $\mathbf{\hat{\theta}} = (\mathbf{X}^{\intercal}\mathbf{X})^{-1}\mathbf{X}^{\intercal}\mathbf{y}$
            *   Complexity: $O(n^{2.4})$ to $O(n^3)$ (scales poorly with features $n$)
        *   **B. Gradient Descent (Iterative)**
            *   Goal: Minimize cost function by moving along the negative gradient
            *   Key Hyperparameter: Learning Rate ($\eta$)
            *   **Variants:**
                *   Batch GD (BGD): Uses all $m$ instances; slow but precise convergence
                *   Stochastic GD (SGD): Uses 1 instance; fast but erratic; good for escaping local minima; requires learning schedule
                *   Mini-batch GD (MBGD): Uses mini-batches; efficient, especially with GPUs

2.  **POLYNOMIAL REGRESSION & MODEL ASSESSMENT**
    *   **Concept:** Uses a linear model trained on polynomial features ($x^2, x^3$)
    *   **Generalization Error Assessment:**
        *   Learning Curves: Plot performance vs. training set size
        *   Overfitting: Low training error, high validation error; large gap
        *   Underfitting: Both errors high and close; model is too simple

3.  **REGULARIZED LINEAR MODELS (Fighting Overfitting)**
    *   **Constraint:** Limits the magnitude of model weights
    *   **Ridge Regression:** $L_2$ penalty; weights shrink towards zero
    *   **Lasso Regression:** $L_1$ penalty; performs automatic feature selection (sparse model)
    *   **Elastic Net:** Hybrid of $L_1$ and $L_2$; preferred when features are correlated

4.  **LOGISTIC REGRESSION (Binary Classification)**
    *   **Output:** Estimated probability $p$
    *   **Activation Function:** Logistic (Sigmoid) function $\sigma(z)$
    *   **Cost Function:** Log Loss (Cross Entropy); convex
    *   **Decision Boundary:** Linear boundary at $p=0.5$

5.  **SOFTMAX REGRESSION (Multiclass Classification)**
    *   **Output:** Probability $p_k$ for each class $k$
    *   **Activation Function:** Softmax function
    *   **Cost Function:** Cross Entropy
    *   **Decision Boundary:** Linear boundaries between classes

***