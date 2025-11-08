## Chapter 6: Decision Trees – Comprehensive Guide

Decision Trees (DTs) are versatile Machine Learning algorithms capable of performing both classification and regression tasks, as well as multioutput tasks. They form the fundamental components of powerful algorithms like Random Forests.

### I. Core Strengths and Interpretation

| Feature | Description | Source |
| :--- | :--- | :--- |
| **Versatility** | DTs can handle both classification and regression tasks. | |
| **Data Preparation** | They require very little data preparation; specifically, they do not require feature scaling or centering. | |
| **Model Type** | They are often called **white box models** because their decisions are intuitive and their rules are easy to interpret, unlike complex models like Random Forests or neural networks (black box models). | |
| **Applications** | Decision Tree Regression was used in the California housing project, where it fit the training data perfectly (though it tended to overfit). | |

### II. Training and Prediction

Training a Decision Tree involves using a specific implementation, like the `DecisionTreeClassifier` (for classification) or `DecisionTreeRegressor` (for regression) in Scikit-Learn.

#### A. Making Predictions (Classification)

To classify an instance, you start at the root node and traverse the tree based on the feature comparison at each node.

| Node Attribute | Purpose | Source |
| :--- | :--- | :--- |
| `samples` | Counts how many training instances the node applies to. | |
| `value` | Shows how many training instances of each class this node applies to (e.g., for classes 0, 1, and 2). | |
| `gini` | Measures the node's **impurity**. A node is "pure" if all instances belong to the same class (gini=0). | |

#### B. Estimating Class Probabilities

A Decision Tree can estimate the probability that an instance belongs to a specific class $k$:
1. It traverses the tree to find the corresponding leaf node for the instance.
2. It returns the ratio of training instances of class $k$ found within that leaf node.
3. The predicted class is the one with the highest estimated probability.

### III. The CART Training Algorithm

Scikit-Learn uses the **Classification and Regression Tree (CART)** algorithm to train (or "grow") Decision Trees.

| Concept | Details | Source |
| :--- | :--- | :--- |
| **Mechanism** | CART is a **greedy algorithm**; it searches for an optimum split at the current level without checking if that split leads to the lowest impurity several levels down. | |
| **Structure** | CART produces only **binary trees**; nonleaf nodes always have two children (yes/no answers). | |
| **Objective** | It searches for the feature $k$ and threshold $t_k$ that produce the purest subsets, weighted by their size. | |
| **Optimality** | Finding the optimal tree is known to be an **NP-Complete problem**, which is why greedy algorithms are used to find a "reasonably good" solution instead. | |

#### A. Cost Functions

| Task | Objective | Cost Function | Source |
| :--- | :--- | :--- | :--- |
| **Classification** | Minimize impurity (Gini or Entropy). $G_i = 1 - \sum_{k=1}^n p_{i, k}^2$ where $p_{i,k}$ is the ratio of class $k$ instances in the node $i$. | $J(k, t_k) = \frac{m_{left}}{m} G_{left} + \frac{m_{right}}{m} G_{right}$ | |
| **Regression** | Minimize the Mean Squared Error (MSE). | $J(k, t_k) = \frac{m_{left}}{m} \text{MSE}_{left} + \frac{m_{right}}{m} \text{MSE}_{right}$ | |

#### B. Gini Impurity vs. Entropy

Entropy ($H_i$) is an alternative impurity measure.
$$H_i = - \sum_{\substack{k=1 \\ p_{i, k} \ne 0}}^n p_{i, k} \log_2 p_{i, k}$$
**Comparison:**
*   **Most of the time**, they lead to similar trees.
*   **Gini Impurity** is slightly **faster to compute** (good default) and tends to isolate the most frequent class in its own branch.
*   **Entropy** tends to produce slightly **more balanced trees**.

### IV. Computational and Stability

| Aspect | Detail | Source |
| :--- | :--- | :--- |
| **Computational Complexity (Training)** | $O(n \times m \log(m))$, where $n$ is the number of features and $m$ is the number of instances. If the training set size increases by a factor of 10, the training time increases by roughly $\approx 11.7$ times (assuming $m=10^6$). | |
| **Instability** | Decision Trees love **orthogonal decision boundaries** (splits perpendicular to an axis), making them highly sensitive to training set rotation. | |
| **Sensitivity** | Small variations in the training data (e.g., removing a few instances) can lead to dramatically different Decision Trees (high variance). | |

### V. Regularization and Preventing Overfitting

Decision Trees are prone to overfitting if trained without restrictions.

| Technique | Hyperparameter(s) | Effect | Source |
| :--- | :--- | :--- | :--- |
| **Maximum Depth** | `max_depth` | Decreasing this constrains the model and regularizes it. | |
| **Minimum Node Size** | `min_samples_split`, `min_samples_leaf` | Controls the minimum number of samples a node must contain to split, or the minimum number in a leaf node. | |
| **Pruning** | Statistical tests (e.g., $\chi^2$ test) | Works by training the tree without restrictions, then deleting unnecessary nodes whose purity improvement is not statistically significant. | |

**Note on Depth:** The depth of an unrestricted Decision Tree trained on $m$ instances is approximately $\log_2(m)$ (e.g., $\approx 20$ for one million instances).

***

## Mind Map Outline for Chapter 6: Decision Trees

This outline provides a structured overview of the material for quick reference.

### 1. Introduction & Core Features
*   1.1. Versatility (Classification & Regression)
*   1.2. Low Data Prep (No Scaling/Centering needed)
*   1.3. Model Type: White Box (Intuitive rules)
*   1.4. Instability (High variance)
    *   1.4.1. Sensitive to rotation
    *   1.4.2. Sensitive to small data changes

### 2. Training (CART Algorithm)
*   2.1. Scikit-Learn Classifiers (`DecisionTreeClassifier`)
*   2.2. Scikit-Learn Regressors (`DecisionTreeRegressor`)
*   2.3. Mechanism: Greedy search for optimal split
*   2.4. Output: Binary Trees
*   2.5. Computational Complexity: $O(n \times m \log(m))$

### 3. Classification Details
*   3.1. Prediction Path: Start at root $\to$ traverse nodes
*   3.2. Node Attributes: `samples`, `value`, `gini`
*   3.3. Impurity Measures
    *   3.3.1. Gini Impurity (Faster, Default)
    *   3.3.2. Entropy ($H_i$) (Balanced trees)
*   3.4. Probabilities: Class ratio in leaf node (`predict_proba`)

### 4. Regression Details
*   4.1. Prediction Value: Average target value in leaf
*   4.2. Cost Function: Minimize MSE
*   4.3. Overfitting: Highly prone, requires regularization

### 5. Regularization (Controlling Overfitting)
*   5.1. Stopping Conditions
    *   5.1.1. `max_depth`
    *   5.1.2. `min_samples_leaf` (most important)
    *   5.1.3. `min_samples_split`
*   5.2. Pruning (Post-training deletion of unnecessary nodes)

***