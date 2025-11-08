## Chapter 9: Unsupervised Learning Techniques

The objective of unsupervised learning (UL) is to find patterns in data that lacks labels. This field holds vast potential, often referred to as the "cake" of intelligence. Chapter 9 focuses on key UL tasks beyond dimensionality reduction (covered in Chapter 8).

### I. Core Unsupervised Learning Tasks

1.  **Clustering:** The goal is to group similar instances together into clusters.
    *   **Applications:** Data analysis, customer segmentation (useful in recommender systems), search engines, image segmentation, semi-supervised learning, and dimensionality reduction.
    *   **Types:** *Hard clustering* assigns each instance to a single cluster, while *soft clustering* provides a score (distance or similarity) per cluster.
2.  **Anomaly Detection:** The system learns what "normal" data looks like to detect abnormal or outlier instances (e.g., defective items on a production line).
    *   **Related Task:** *Novelty detection* is a closely related task where the algorithm is trained on a "clean" dataset (uncontaminated by outliers).
3.  **Density Estimation:** The objective is to estimate the probability density function (PDF) of the training set. This is useful for anomaly detection and data analysis.

### II. K-Means Clustering

K-Means is a popular, fast, and scalable clustering algorithm. It aims to find the centers (centroids) of $k$ predefined clusters.

#### A. Algorithm and Implementation
*   **Concept:** The algorithm iteratively labels instances by assigning them to the closest centroid and then updates the centroids by calculating the mean of all instances assigned to that cluster.
*   **Convergence:** The algorithm is guaranteed to converge in a finite (usually small) number of steps because the mean squared distance between instances and their closest centroid can only decrease at each step.
*   **Initialization Issue:** K-Means is sensitive to random initialization and can converge to a suboptimal solution (non-global minimum). To mitigate this, set the `n_init` hyperparameter to a value greater than 1 (e.g., 10), which runs the algorithm multiple times and keeps the best result.
*   **Soft Clustering:** The `KMeans` class offers a `transform()` method, which measures the distance of each instance to every centroid, providing a form of soft clustering.
*   **Limitations:** Performs poorly when clusters have highly unequal sizes, different densities, or non-spherical shapes.

#### B. Selecting the Optimal Number of Clusters ($k$)
The performance measure typically minimized during K-Means training is **Inertia**, the mean squared distance between each instance and its closest centroid. However, inertia is not suitable for choosing $k$, as it continuously decreases when $k$ increases.

1.  **Elbow Method:** Plot the inertia as a function of $k$ and select the value where the decrease rate dramatically slows down (the "elbow").
2.  **Silhouette Score:** The silhouette coefficient for an instance is calculated as $\frac{b - a}{\max(a, b)}$:
    *   $a$: Mean distance to the other instances in the *same* cluster (mean intra-cluster distance).
    *   $b$: Mean distance to the instances in the *next closest* cluster (mean nearest-cluster distance).
    *   The Silhouette Score is the mean of all these coefficients. Scores close to +1 are good, 0 means the instance is near a decision boundary, and -1 means it was likely assigned to the wrong cluster.
3.  **Silhouette Diagram:** A visual plot showing each instance's silhouette coefficient, sorted by cluster and coefficient value. Clusters are considered poor if many instances fall short of the mean silhouette score (represented by a dashed line).

### III. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is a powerful algorithm that defines clusters as continuous regions of high-density instances.

*   **Instance Types:** DBSCAN identifies three types of instances:
    1.  **Core Instances:** Instances that have a minimum number of neighbors ($\text{min\_samples}$) within a distance $\epsilon$ (epsilon).
    2.  **Border Instances:** Instances whose number of neighbors is below $\text{min\_samples}$, but lie within the $\epsilon$ radius of a core instance.
    3.  **Anomalies (Outliers):** All other instances.
*   **Strengths:** Can identify any number of clusters of arbitrary shape, and is robust to outliers.
*   **Weaknesses:** Cannot capture all clusters properly if the density varies significantly across them.
*   **Prediction for New Data:** The `DBSCAN` class lacks a `predict()` method. To predict cluster membership for new instances, a separate classifier (e.g., `KNeighborsClassifier`) must be trained on the identified core instances.
*   **Complexity:** Computational complexity is roughly $O(m \log m)$, making it nearly linear, but the Scikit-Learn implementation can require up to $O(m^2)$ memory if $\epsilon$ is large.

### IV. Gaussian Mixture Models (GMM)

GMMs are *generative models* that assume instances are generated from a mixture of several Gaussian (Normal) probability distributions.

#### A. Architecture and Training
*   **Generative Process:** Each instance $x^{(i)}$ is assumed to belong to a hidden (latent) cluster $z^{(i)}$.
*   **Parameters:** GMMs learn the parameters for each Gaussian distribution: the mean ($\mu^{(k)}$), the covariance matrix ($\Sigma^{(k)}$), and the overall cluster weights ($\phi^{(k)}$).
*   **Training Method:** Uses the **Expectation-Maximization (EM) algorithm**.
    *   **E-Step (Expectation):** Assigns instances to clusters based on current parameter estimates (soft assignment via probability estimation).
    *   **M-Step (Maximization):** Updates parameters based on the new cluster assignments, maximizing the likelihood function.
*   **Initialization:** Like K-Means, EM can converge to poor local solutions; setting `n_init` (e.g., to 10) helps find a better solution.
*   **Covariance Constraints:** Constraints can be imposed on the covariance matrices using the `covariance_type` hyperparameter:
    *   `"full"` (default): Each cluster has its own unconstrained covariance matrix (any shape, size, orientation).
    *   `"tied"`: All clusters share the same covariance matrix.
    *   `"diag"`: Ellipsoid axes must be parallel to coordinate axes (diagonal covariance matrices).
    *   `"spherical"`: All clusters must be spherical, but can have different diameters (variances).
*   **Anomaly Detection:** GMMs are highly effective for density estimation. Anomalies can be detected by setting a density threshold (e.g., instances below the 4th percentile density).

#### B. Model Selection for GMM
To select the optimal number of clusters ($k$) and `covariance_type`, theoretical information criteria are typically used, as they penalize complexity.

1.  **Bayesian Information Criterion (BIC):** Tends to select a simpler model.
    $$\text{BIC} = \log(m)p - 2 \log(L) \text{}$$
2.  **Akaike Information Criterion (AIC):**
    $$\text{AIC} = 2p - 2 \log(L) \text{}$$
    *   $m$: number of instances.
    *   $p$: number of parameters learned by the model.
    *   $L$: maximized value of the likelihood function.
    The best model is generally the one that minimizes the BIC or AIC.

3.  **Bayesian Gaussian Mixture Models:** The `BayesianGaussianMixture` class automatically selects the effective number of clusters by setting weights of unnecessary clusters close to zero. The `weight_concentration_prior` hyperparameter controls the prior belief about the number of clusters (low value = few clusters; high value = plentiful clusters).

### V. Other Unsupervised Techniques (Overview)

| Algorithm | Key Features | Primary Use |
| :--- | :--- | :--- |
| **Mini-batch K-Means** | Faster convergence than standard K-Means by using mini-batches (random subsets of data) to move centroids. | Scalable for large datasets. |
| **Agglomerative Clustering** | A *hierarchical clustering* algorithm. Starts with individual instances and merges the closest pairs/clusters. | Produces an informative cluster tree. Scalable for large datasets if a *connectivity matrix* is provided. |
| **Mean-Shift** | Locates and shifts toward regions of high density (attractors). Does not require specifying $k$. | Finding clusters of various shapes/densities. |
| **Affinity Propagation** | Uses a messaging mechanism where instances communicate similarity until convergence. Does not require specifying $k$. | Finds clusters quickly based on similarity. |
| **Spectral Clustering** | Uses the similarity matrix between instances and often relies on dimensionality reduction (e.g., Kernel PCA). | Complex clustering tasks. |
| **Local Outlier Factor (LOF)** | Compares density around an instance to density around its neighbors. | Outlier detection. |
| **One-Class SVM** | Finds a small region in high-dimensional space that encompasses all training instances. | Novelty detection (trained on clean, outlier-free data). |

***

## Mind Map: Unsupervised Learning Techniques (Chapter 9)

**ROOT: Unsupervised Learning Techniques**

1.  **Core Tasks**
    *   Clustering: Grouping similar instances.
    *   Anomaly Detection: Detecting abnormal instances.
    *   Density Estimation: Estimating PDF.

2.  **Clustering Algorithms**
    *   **K-Means**
        *   Concept: Finds $k$ centroids; instance assigned to closest centroid.
        *   Output: Hard Clustering (instance $\to$ single cluster).
        *   Limitations: Sensitive to initialization; poor with unequal size/density/non-spherical clusters.
        *   Choosing $k$ (Model Selection):
            *   Inertia (Error Metric).
            *   Silhouette Score / Diagram.
    *   **DBSCAN**
        *   Concept: Clusters high-density regions.
        *   Output: Core, Border, and Anomaly Instances.
        *   Strengths: Finds arbitrary shapes; robust to outliers.
        *   Prediction: Requires training separate classifier (e.g., KNN) on core instances.
    *   **Gaussian Mixture Model (GMM)**
        *   Concept: Generative model assuming data from Gaussian distribution mixture.
        *   Training: Expectation-Maximization (EM) algorithm.
        *   Constraints: `covariance_type` (full, tied, diag, spherical).
        *   Model Selection: BIC / AIC (penalize model complexity).
        *   Bayesian GMM: Automatically selects $k$ (unnecessary clusters $\to$ zero weight).
    *   **Other Clustering**
        *   Mini-batch K-Means (Scaling).
        *   Agglomerative (Hierarchical).
        *   Mean-Shift / Affinity Propagation / Spectral Clustering.

3.  **Advanced Topics / Applications**
    *   **Semi-supervised Learning**
    *   **Image Segmentation** (Color Quantization).
    *   **Anomaly/Novelty Detection**
        *   LOF (Outlier detection).
        *   One-Class SVM (Novelty detection on clean datasets).