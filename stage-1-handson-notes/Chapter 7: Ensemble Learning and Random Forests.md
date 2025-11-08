## Chapter 7: Ensemble Learning and Random Forests

Ensemble Learning is a technique that involves aggregating the predictions of a group of predictors (such as classifiers or regressors). This aggregation often results in predictions that are superior to those made by the best individual predictor.

*   A group of predictors is called an **ensemble**.
*   An Ensemble Learning algorithm is called an **Ensemble method**.
*   Ensemble methods are typically used late in a project to combine already built predictors into an even better predictor.

---

### I. Voting Classifiers

A simple way to create a better classifier is to aggregate the predictions of several individually trained classifiers.

#### 1. Hard Voting Classifier

*   **Mechanism:** Predicts the class that receives the most votes (majority class) from the individual classifiers.
*   **Performance:** A voting classifier often achieves higher accuracy than the best individual classifier in the ensemble.
*   **Strong Learners from Weak Learners:** Even if individual classifiers are only **weak learners** (performing only slightly better than random guessing), the ensemble can become a **strong learner** (achieving high accuracy), provided there is a sufficient number of weak learners that are sufficiently diverse.
*   **Diversity Requirement:** This improvement relies on the classifiers making uncorrelated errors.

#### 2. Soft Voting Classifier

*   **Mechanism:** If individual classifiers can estimate class probabilities, the ensemble averages these probabilities and predicts the class with the highest average probability.
*   **Performance:** Soft voting often achieves better performance than hard voting, as it gives more weight to confident predictions.

---

### II. Bagging and Pasting

These methods generate diverse predictors by training each predictor on a different random subset of the training set.

*   **Training Process:** Predictors are trained in parallel.
*   **Prediction Aggregation:** Typically uses soft or hard voting for classification, or averaging for regression.

| Technique | Sampling Method | Description |
| :--- | :--- | :--- |
| **Bagging** (Bootstrap Aggregating) | Sampling **with** replacement | Allows training instances to be sampled several times for the same predictor. |
| **Pasting** | Sampling **without** replacement | Training instances are sampled only once for the entire ensemble. |

#### 1. Benefits (Bias-Variance Trade-off)

*   Bagging and pasting lead to predictions that generalize much better than a single predictor.
*   The ensemble has a comparable bias but a smaller variance (less irregular decision boundary).

#### 2. Out-of-Bag (OOB) Evaluation (Bagging Only)

*   **OOB Instances:** With bagging, some instances in the training set may never be sampled for a specific predictor. These instances are called out-of-bag instances.
*   **Validation:** OOB instances can be used as a free validation set, eliminating the need for a separate validation set. For each instance, the ensemble prediction is the aggregation of predictions from only the predictors that never saw it during training.

#### 3. Random Patches and Random Subspaces

These are variations where features are also sampled, not just instances.

*   **Random Patches:** Sampling of both training instances and features.
*   **Random Subspaces:** Sampling of features only (keeping all training instances).

---

### III. Random Forests

A Random Forest is an ensemble of Decision Trees. It is one of the most powerful Machine Learning algorithms available today.

*   **Prediction:** Predictions are made by obtaining the predictions of all individual trees and predicting the class that receives the most votes (hard voting).
*   **Extra Randomness:** Random Forests add extra randomness to Decision Trees by searching for the best feature among a random subset of features (instead of searching across all features) when splitting a node. This greater diversity results in lower variance.

#### Feature Importance

*   Random Forests are useful for gaining a quick understanding of which features matter most.
*   They can indicate the relative importance of each attribute for accurate predictions. This is helpful if you need to perform feature selection.

#### Extremely Randomized Trees (Extra-Trees)

This is an even more random ensemble of Decision Trees. Instead of searching for the best possible thresholds for each feature (like a Random Forest), it uses random thresholds for every feature for maximum diversity. This trades slightly higher bias for lower variance and significantly speeds up training.

---

### IV. Boosting

Boosting methods combine several weak learners sequentially into a strong learner. Each successive predictor tries to correct the errors of its predecessor.

#### 1. AdaBoost (Adaptive Boosting)

*   **Mechanism:** Focuses on misclassified training instances. The training algorithm assigns high relative weights to instances that the previous predictors got wrong.
*   **Prediction:** The ensemble uses a weighted majority vote, where predictors are assigned different weights based on their overall accuracy on the weighted training set.
*   **Sequential Limitation:** This sequential learning technique cannot be parallelized easily, so it does not scale as well as bagging or pasting.

#### 2. Gradient Boosting

*   **Mechanism:** Works by training new predictors sequentially, where each new predictor is trained on the residual errors (or residuals) made by the previous predictor.
*   **Shrinkage (Regularization):** The `learning_rate` hyperparameter scales the contribution of each tree. Setting it to a low value (e.g., 0.1) requires more trees but usually leads to better generalization. This is a regularization technique.
*   **Early Stopping:** To find the optimal number of trees (and prevent overfitting), training should be stopped when the validation error stops improving. This can be implemented by setting `warm_start=True` to allow incremental training.

---

### V. Stacking (Stacked Generalization)

Stacking is an Ensemble method where a final predictor (called a **blender** or **meta-learner**) is used to aggregate the predictions of all the individual predictors.

*   **Process:**
    1.  The base predictors in the first layer are trained on the original training set.
    2.  The final predictor (blender) is trained on the predictions output by the first layer predictors.
*   **Training the Blender:**
    *   **Blending:** Using a **hold-out set** (a separate validation set) to train the blender.
    *   **Stacking (Out-of-Fold):** Using cross-validation to generate "out-of-fold" predictions for training the blender.
*   **Multi-Layer Stacking:** Stacking can involve multiple layers of blenders, where a second layer blender takes the first layer blenders' predictions as input.
*   **Implementation:** Scikit-Learn does not directly support stacking, but specialized libraries like DESlib or custom implementations can be used.

---

## Mind Map: Chapter 7 Overview

```
1. ENSEMBLE LEARNING (Aggregation)
    ├── Goal: Reduce error and improve generalization
    ├── Predictors are diverse (make uncorrelated errors)
    
2. VOTING CLASSIFIERS
    ├── Hard Voting (Majority Class)
    ├── Soft Voting (Averaged Probabilities)
    
3. PARALLEL ENSEMBLES (Randomized Data/Features)
    ├── Bagging (Bootstrap Aggregating)
    │   ├── Sampling with replacement
    │   └── Out-of-Bag (OOB) Evaluation (Free validation)
    ├── Pasting
    │   └── Sampling without replacement
    ├── Random Patches/Subspaces
    
4. RANDOM FORESTS
    ├── Ensemble of Decision Trees (via Bagging typically)
    ├── Key Feature: Random subset of features considered at each split
    └── Use Case: Feature Importance estimation
    
5. SEQUENTIAL ENSEMBLES (Correcting Errors)
    ├── AdaBoost (Adaptive Boosting)
    │   └── Focus: Boost weights of misclassified instances
    └── Gradient Boosting (GBRT)
        ├── Focus: Train successors on predecessor's residual errors
        ├── Techniques: Shrinkage (Learning Rate) & Early Stopping
        
6. STACKING (Stacked Generalization)
    ├── Structure: Multi-layer aggregation
    └── Blender (Meta-learner)
        ├── Trained on predictions of first-layer predictors
        └── Blending uses a hold-out set
```