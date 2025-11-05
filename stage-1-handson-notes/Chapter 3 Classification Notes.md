# **Chapter 3: Classification Notes**  
Classification is the supervised task of predicting discrete classes.  
## **I. Binary Classification and Performance Metrics**  
**1. MNIST Dataset**  
* **Source:** 70,000 small images of handwritten digits.  
* **Format:** Each image is $28 \times 28$ pixels, represented by 784 features (pixel intensities from 0 to 255).  
* **Setup:** The dataset is typically split into a training set (first 60,000 images) and a test set (last 10,000).  
* **Binary Task Example:** Training a "5-detector" to distinguish between digit 5 and "not-5".  
* **Algorithm Example:** The **Stochastic Gradient Descent (SGD) classifier** is often a good starting point, as it handles large datasets efficiently and is suitable for online learning.  
**2. Evaluating Classifiers**  
Evaluating classifiers is complex, especially on **skewed datasets** (where some classes are much rarer than others).  

| Metric | Calculation | Purpose/Context |
| ---------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Accuracy | Ratio of correct predictions. | Insufficient for skewed datasets (e.g., guessing "not-5" still yields >90% accuracy on MNIST). |
| Cross-Validation | Typically K-fold or using StratifiedKFold to ensure representative ratios of each class in folds. | Provides an estimate of performance and a measure of its precision (standard deviation). |
| Confusion Matrix | Counts how many times class A is classified as class B. | Rows = Actual Class; Columns = Predicted Class. |
| Precision | $\\frac{\\text{True Positives (TP)}}{\\text{TP} + \\text{False Positives (FP)}}$ | Accuracy of positive predictions. (Low FP is high Precision). |
| Recall (TPR) | $\\frac{\\text{TP}}{\\text{TP} + \\text{False Negatives (FN)}}$ | Ratio of positive instances correctly detected. (Low FN is high Recall). |
| F1 Score | Harmonic mean of Precision and Recall. | Favors classifiers with similar Precision and Recall; useful for comparing models. |
  
**3. Precision/Recall Trade-off & ROC Curves**  
A classifier relies on a **decision function** that outputs a score, which is then compared against a **threshold** to make a positive or negative prediction.  
* **Trade-off:** Increasing the threshold **increases Precision** but **decreases Recall**, and vice versa.  
* **Visualization:** Plot Precision against Recall (PR curve) or plot both against the threshold value.  
* **Decision Scores:** You can access underlying decision scores using cross_val_predict(..., method="decision_function") or probabilities using predict_proba() to select a custom threshold.  
* **ROC Curve (Receiver Operating Characteristic):**  
    * Plots the **True Positive Rate (Recall)** against the **False Positive Rate (FPR)**.  
    * $\text{FPR} = 1 - \text{Specificity}$ (where Specificity is the True Negative Rate).  
    * **AUC (Area Under the Curve):** Measures classifier performance. $AUC=1$ is perfect; $AUC=0.5$ is random.  
* **PR vs. ROC:** Prefer the **PR curve** when the positive class is rare or when False Positives are more critical than False Negatives. Otherwise, use the **ROC curve**.  
## **II. Advanced Classification Types**  
**1. Multiclass Classification**  
Distinguishing between more than two classes (e.g., digits 0 through 9).  

| Strategy | Description | When to Use |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| One-versus-the-rest (OvR) | Trains one binary classifier for each class. Chooses the class with the highest score. | Preferred for most algorithms. |
| One-versus-one (OvO) | Trains a binary classifier for every pair of classes ($N(N-1)/2$). Chooses the class that wins the most duels. | Preferred for algorithms that scale poorly with the size of the training set (e.g., Support Vector Machines). |
| Native | Algorithms like SGD classifiers, Random Forest classifiers, and naive Bayes classifiers inherently handle multiple classes. | Automatically used by Scikit-Learn. |
  
**	•	Data Scaling:** Scaling inputs (e.g., using StandardScaler) can significantly increase accuracy in multiclass tasks.  
**2. Error Analysis**  
Analyze the **confusion matrix** to understand the type of errors the classifier makes, which provides insights into model improvement.  
1. Compute the confusion matrix (e.g., using confusion_matrix(y_train, y_train_pred)).  
2. Normalize the matrix by dividing each value by the number of instances in the *actual class* (row sums) to compare error rates fairly across classes.  
3. Visualization (e.g., using Matplotlib's matshow()) makes patterns obvious, such as high misclassification of a certain digit, guiding efforts toward adding features or preprocessing.  
**3. Multilabel and Multioutput Classification**  
* **Multilabel Classification:** A system that outputs multiple binary tags for a single instance (e.g., detecting multiple people in one photo). Evaluation often uses the average F1 score across all labels (e.g., average="macro" or average="weighted").  
* **Multioutput Classification:** A generalization of multilabel classification where each label can be multiclass (i.e., having more than two possible values). Example: Denoising an image, where each pixel (label) can take one of 256 intensity values.  
## **III. Classification Mind Map (Visual Outline)**  
```
mindmap
  root((Chapter 3: Classification))
    1. MNIST & Binary Classifiers
      Images of Digits (70,000)
      Features (784)
      Classifier: SGD (Stochastic Gradient Descent)

    2. Performance Measures
      Accuracy
        Ineffective for skewed datasets
      Confusion Matrix
        TP, TN, FP, FN
        Used for Error Analysis
      Metrics derived from CM
        Precision: TP / (TP + FP)
        Recall (TPR / Sensitivity): TP / (TP + FN)
        F1 Score (Harmonic Mean)
      Trade-off: Precision/Recall
        Decision Function & Threshold
        Raise Threshold -> High Precision, Low Recall
      ROC Curve
        Plot: Recall vs. FPR (False Positive Rate)
        Area Under Curve (AUC)
        Rule: Use PR curve if positive class is rare

    3. Multiclass
      Strategies
        One-vs-Rest (OvR)
        One-vs-One (OvO)
      Error Analysis
        Normalize Confusion Matrix by row
        Find systematic misclassifications
      Scaling Input: Important for accuracy

    4. Advanced Tasks
      Multilabel Classification
        Multiple binary tags per instance
        Evaluation: Averaged F1 score
      Multioutput Classification
        Generalization of Multilabel
        Each output label is multiclass (e.g., Denoising)

```
