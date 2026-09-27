# ACADEMIC PROJECT REPORT

## Explainable AI-Based Rockfall Risk Prediction and Early Warning System for Open-Pit Mines

**A Project Report submitted in partial fulfillment of the requirements for the degree of Bachelor of Engineering in Computer Science and Engineering**

---

### ABSTRACT
Slope instability and unexpected rockfalls in open-pit mining operations represent catastrophic geotechnical hazards capable of causing loss of human life, destruction of heavy extraction machinery, and severe production bottlenecks. Traditional limit-equilibrium methods (LEM) and finite-element modeling (FEM) require rigorous site parameters and extensive computational duration, limiting their applicability for real-time hazard screening. This project presents an Explainable Artificial Intelligence (XAI) framework for predicting slope failure risk using a comprehensive 10,000-sample slope stability dataset. 

A rigorous, zero-leakage scikit-learn preprocessing pipeline was engineered with standardized numerical scaling and one-hot categorical encoding. Four classification architectures were trained and empirically evaluated: Logistic Regression, Support Vector Machine (SVM) with calibrated RBF kernel, Random Forest, and Extreme Gradient Boosting (XGBoost). Given the pronounced class imbalance (2.95% unstable cases, $FS < 1.0$), models were evaluated on hold-out test data prioritizing F1-score and Recall. SVM achieved superior screening performance with an F1-score of 0.9027, Recall of 86.44%, Precision of 94.44%, Accuracy of 99.45%, and ROC-AUC of 0.9983, effectively suppressing false alarms while detecting critical hazards.

To resolve the "black-box" nature of nonlinear models, SHAP (SHapley Additive exPlanations) was integrated to provide both global feature importance rankings and local sample-level attribution waterfalls. An interactive multi-page web application was engineered in Streamlit, featuring dynamic inference, automated early-warning status indicators, what-if sensitivity analysis, and dataset exploration.

---

### TABLE OF CONTENTS
1. **Introduction**
   * 1.1 Background and Motivation
   * 1.2 Problem Statement
   * 1.3 Project Objectives
   * 1.4 Scope and Limitations
2. **Geotechnical Concepts & Theoretical Background**
   * 2.1 Factor of Safety (FS) Mechanics
   * 2.2 Mohr-Coulomb Failure Criterion
   * 2.3 Influence of Groundwater Pore Pressure ($r_u$)
   * 2.4 Engineered Reinforcement Measures
3. **Dataset Architecture & Preprocessing**
   * 3.1 Dataset Exploration & Schema
   * 3.2 Target Variable Formulation
   * 3.3 Zero-Leakage Preprocessing Pipeline
   * 3.4 Stratified Train/Test Partitioning
4. **Machine Learning Methodology**
   * 4.1 Logistic Regression (Baseline)
   * 4.2 Support Vector Machine (SVM)
   * 4.3 Random Forest Classifier
   * 4.4 XGBoost (Extreme Gradient Boosting)
5. **Explainable AI (XAI) Framework**
   * 5.1 Game-Theoretic Foundations of SHAP
   * 5.2 Global Feature Importance
   * 5.3 Local Instance-Level Attribution
6. **Empirical Results & Comparative Analysis**
   * 6.1 Hold-Out Test Metrics
   * 6.2 Confusion Matrix Evaluation
   * 6.3 ROC-AUC Characteristics
   * 6.4 Model Selection Discussion
7. **System Architecture & Application Implementation**
   * 7.1 Pipeline Modularization
   * 7.2 Streamlit User Interface
   * 7.3 What-If Sensitivity Simulator
8. **Conclusion & Future Directions**
   * 8.1 Summary of Contributions
   * 8.2 Future Scope
9. **References**

---

### CHAPTER 1: INTRODUCTION

#### 1.1 Background and Motivation
In surface open-pit mining, bench walls are excavated at steep angles to maximize ore extraction while minimizing waste stripping ratios. However, steep slopes are susceptible to shear failure along continuous discontinuity planes or weak rock mass zones. Slope failures often trigger sudden rockfalls that hazard haul roads, drilling benches, and personnel. Rapid screening tools can assist geotechnical engineers in prioritizing inspections and dispatching mitigation measures.

#### 1.2 Problem Statement
Traditional geotechnical stability evaluations rely on Limit Equilibrium Methods (LEM), such as Bishop’s Simplified or Spencer’s methods, which require constructing complex geometric cross-sections and iterating failure slip surfaces. In dynamic mining settings, these analyses cannot be performed interactively for hundreds of bench sections. A data-driven machine learning system capable of rapid, accurate classification while remaining interpretable is essential.

#### 1.3 Project Objectives
1. **Data Pipeline Engineering:** Build a clean, zero-leakage data transformation pipeline handling continuous physical parameters and categorical support types.
2. **Multi-Model Benchmark:** Train, tune, and evaluate four diverse classifiers (Logistic Regression, SVM, Random Forest, XGBoost).
3. **Imbalance-Aware Optimization:** Address the 33:1 negative-to-positive class imbalance using cost-sensitive weighting and stratified sampling.
4. **Explainability Integration:** Formulate SHAP explanations for global importance and local risk predictions.
5. **Interactive Prototype Deployment:** Develop a Streamlit application featuring live inference, what-if sensitivity analysis, and early-warning alerts.

#### 1.4 Scope and Limitations
* **Educational Prototype:** The system utilizes a standardized slope stability dataset as a proxy for rockfall hazard screening. It is not trained on verified historical rockfall disaster records or live operational telemetry.
* **Non-Causal Statistical Engine:** Predictions and SHAP attributions reflect statistical associations within the trained model and do not replace certified on-site geotechnical instrumentation.

---

### CHAPTER 2: GEOTECHNICAL CONCEPTS & THEORETICAL BACKGROUND

#### 2.1 Factor of Safety (FS) Mechanics
The Factor of Safety ($FS$) is the ratio of available shear strength ($\tau_f$) along a prospective slip plane to the mobilized shear stress ($\tau$) required for equilibrium:
$$FS = \frac{\tau_f}{\tau}$$

* **$FS \ge 1.0$ (Stable):** Resisting forces exceed driving forces.
* **$FS < 1.0$ (Unstable / Hazard):** Driving gravitational and water forces exceed resisting shear strength, leading to slope failure.

#### 2.2 Mohr-Coulomb Failure Criterion
The shear strength of rock and soil is governed by the Mohr-Coulomb relationship:
$$\tau_f = c + (\sigma_n - u) \tan\phi$$
Where:
* $c$: Cohesion ($\text{kPa}$)
* $\sigma_n$: Total normal stress on the failure surface
* $u$: Groundwater pore water pressure
* $\phi$: Internal angle of friction (degrees)

#### 2.3 Influence of Groundwater Pore Pressure ($r_u$)
Groundwater exerts pore pressure $u$, reducing the effective normal stress ($\sigma' = \sigma_n - u$) and lowering shear resistance. The pore water pressure ratio $r_u$ is defined as:
$$r_u = \frac{u}{\gamma \cdot H}$$
Where $\gamma$ is unit weight and $H$ is slope height. Higher $r_u$ values significantly reduce slope stability.

#### 2.4 Engineered Reinforcement Measures
Engineered supports modify slope equilibrium:
1. **Drainage:** Lowers $u$, increasing effective normal stress and resisting strength.
2. **Soil Nailing / Rock Bolts:** Provides passive tensile and shear resistance across slip surfaces.
3. **Retaining Walls:** Provides physical toe restraint, directly counteracting driving moments.
4. **Geosynthetics:** Enhances tensile strength in shallow bench configurations.

---

### CHAPTER 3: DATASET ARCHITECTURE & PREPROCESSING

#### 3.1 Dataset Exploration & Schema
The dataset consists of 10,000 records across 9 attributes:
* `Unit Weight (kN/m³)`: Mean $19.94\,\text{kN/m}^3$ (Range: $15.00 - 25.00$)
* `Cohesion (kPa)`: Mean $27.70\,\text{kPa}$ (Range: $5.01 - 50.00$)
* `Internal Friction Angle (°)`: Mean $32.50^\circ$ (Range: $20.00 - 45.00^\circ$)
* `Slope Angle (°)`: Mean $34.94^\circ$ (Range: $10.00 - 59.99^\circ$)
* `Slope Height (m)`: Mean $27.36\,\text{m}$ (Range: $5.00 - 50.00\,\text{m}$)
* `Pore Water Pressure Ratio`: Mean $0.50$ (Range: $0.00 - 1.00$)
* `Reinforcement Type`: Categorical (`Drainage`, `Geosynthetics`, `Retaining Wall`, `Soil Nailing`)
* `Reinforcement Numeric`: Verified to have a 1-to-1 redundant mapping with Reinforcement Type and excluded from modeling to avoid collinearity.
* `Factor of Safety (FS)`: Mean $2.55$ (Range: $0.50 - 3.00$)

#### 3.2 Target Variable Formulation
$$\text{Risk Target} = (FS < 1.0) \implies \begin{cases} 1 & \text{RISK / UNSTABLE} \quad (295 \text{ samples}, 2.95\%) \\ 0 & \text{STABLE} \quad (9,705 \text{ samples}, 97.05\%) \end{cases}$$

#### 3.3 Zero-Leakage Preprocessing Pipeline
To guarantee that testing metrics reflect true out-of-sample generalization, all transformations were encapsulated in a scikit-learn `ColumnTransformer`:
* **Continuous Features:** Scaled using `StandardScaler` (Z-score normalization: $z = \frac{x - \mu}{\sigma}$), where $\mu$ and $\sigma$ were computed strictly from the training partition.
* **Categorical Feature:** Encoded via `OneHotEncoder(categories=[['Drainage', 'Geosynthetics', 'Retaining Wall', 'Soil Nailing']], handle_unknown='ignore')`.

#### 3.4 Stratified Train/Test Partitioning
The dataset was partitioned using an 80/20 stratified split:
* **Training Partition:** 8,000 samples (7,764 Stable, 236 Unstable)
* **Testing Partition:** 2,000 samples (1,941 Stable, 59 Unstable)
* **Random Seed:** 42 (ensuring full reproducibility)

---

### CHAPTER 4: MACHINE LEARNING METHODOLOGY

#### 4.1 Logistic Regression (Baseline)
Models log-odds of instability as a linear combination of features:
$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \sum_{i=1}^m \beta_i x_i$$
Class weights were inversely proportional to class frequencies: $w_1 = \frac{N}{2 \cdot N_1} \approx 16.95$.

#### 4.2 Support Vector Machine (SVM)
Constructs an optimal separating hyperplane in a high-dimensional feature space using the Radial Basis Function (RBF) kernel:
$$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
Probability calibration was integrated via Platt scaling (`CalibratedClassifierCV(ensemble=False)`), preventing scikit-learn deprecation while providing calibrated posterior probabilities $P(\text{Risk} \mid x)$.

#### 4.3 Random Forest Classifier
An ensemble of 150 bagged decision trees with randomized feature subspace sampling at each split. Balanced subsample weighting penalizes misclassifications of rare unstable instances.

#### 4.4 XGBoost (Extreme Gradient Boosting)
An additive ensemble of decision trees minimizing a regularized log-loss objective:
$$\mathcal{L} = \sum_{i} l(y_i, \hat{y}_i) + \sum_{k} \left(\gamma T_k + \frac{1}{2} \lambda ||w_k||^2\right)$$
Positive instance gradients were amplified by `scale_pos_weight = 32.89`.

---

### CHAPTER 5: EXPLAINABLE AI (XAI) FRAMEWORK

#### 5.1 Game-Theoretic Foundations of SHAP
SHAP computes feature attributions based on classic cooperative game theory (Shapley values):
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$
Where $F$ is the total set of features, $S$ is a feature subset, and $f(S)$ is the model expectation conditioned on $S$.

#### 5.2 Global Feature Importance
Global SHAP analysis revealed the hierarchy of feature influence across the dataset:
1. `Cohesion (kPa)`: Strongest stabilizing factor ($Mean \, |\phi| = 0.162$)
2. `Internal Friction Angle (°)`: Second strongest stabilizing factor ($Mean \, |\phi| = 0.084$)
3. `Slope Angle (°)`: Dominant destabilizing factor ($Mean \, |\phi| = 0.071$)
4. `Pore Water Pressure Ratio`: Dominant fluid driving factor ($Mean \, |\phi| = 0.063$)
5. `Slope Height (m)`: Minor driving factor ($Mean \, |\phi| = 0.012$)
6. `Reinforcement Measures`: Modulating boundary resistance ($Mean \, |\phi| = 0.001 - 0.003$)

#### 5.3 Local Instance-Level Attribution
For an individual slope prediction, SHAP decomposes the prediction into additive contributions:
$$\hat{f}(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
Positive SHAP values represent parameters increasing failure likelihood, while negative values represent parameters reinforcing slope stability.

---

### CHAPTER 6: EMPIRICAL RESULTS & COMPARATIVE ANALYSIS

#### 6.1 Hold-Out Test Metrics
Evaluated on the hold-out test partition of 2,000 samples:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (Selected)** | **0.9945** | **0.9444** | **0.8644** | **0.9027** | **0.9983** |
| **XGBoost** | 0.9915 | 0.8500 | 0.8644 | 0.8571 | 0.9979 |
| **Random Forest** | 0.9900 | 0.8545 | 0.7966 | 0.8246 | 0.9942 |
| **Logistic Regression** | 0.9775 | 0.5673 | 1.0000 | 0.7239 | 0.9999 |

#### 6.2 Confusion Matrix Evaluation
* **SVM:** $TN = 1938, \, FP = 3, \, FN = 8, \, TP = 51$
* **XGBoost:** $TN = 1932, \, FP = 9, \, FN = 8, \, TP = 51$
* **Random Forest:** $TN = 1933, \, FP = 8, \, FN = 12, \, TP = 47$
* **Logistic Regression:** $TN = 1896, \, FP = 45, \, FN = 0, \, TP = 59$

#### 6.3 Discussion of Model Selection
Logistic Regression achieved perfect Recall (1.0000), but produced 45 false positives, lowering Precision to 56.73%. In mining operations, high false alarm rates undermine operator trust and trigger unnecessary, costly bench shutdowns. 

SVM with calibrated RBF kernel achieved the optimal tradeoff:
* Detected **86.44% of critical hazards** (51/59 detected).
* Restricted false alarms to just **3 cases out of 2,000** (Precision: 94.44%).
* Attained the highest overall **F1-score (0.9027)** and highest **Accuracy (99.45%)**.

---

### CHAPTER 7: SYSTEM ARCHITECTURE & APPLICATION IMPLEMENTATION

#### 7.1 Pipeline Modularization
* `src/data_preprocessing.py`: Encapsulates data validation, column harmonization, and the zero-leakage ColumnTransformer.
* `src/train_model.py`: Automates training, benchmark evaluation, model selection, and artifact serialization.
* `src/evaluate_model.py`: Generates standardized metrics, confusion matrices, and ROC data.
* `src/explainability.py`: Houses the SHAP explainability engine supporting Tree and Kernel explainers.

#### 7.2 Streamlit User Interface
The application (`app.py`) provides 8 navigation modules:
1. **Dashboard:** High-level project KPIs and system architecture flowchart.
2. **Risk Prediction:** Dynamic sliders, Predict button, large 🟢 STABLE / 🔴 RISK card, and probability gauge.
3. **Explainable AI:** Global feature rankings and local waterfall attribution plots.
4. **Model Comparison:** Real metric tables, bar charts, confusion matrix heatmaps, and ROC curves.
5. **Risk Analysis:** Factor of Safety distributions, correlation matrices, and boxplots.
6. **What-If Analysis:** Side-by-side scenario simulation with instant probability delta tracking.
7. **Dataset Explorer:** Searchable, filterable 10,000-row table with descriptive statistics.
8. **Project Information:** Viva preparation notes, objectives, technology stack, and academic disclaimer.

---

### CHAPTER 8: CONCLUSION & FUTURE DIRECTIONS

#### 8.1 Summary of Contributions
* Engineered a complete, reproducible machine learning classification system for slope stability assessment.
* Implemented a zero-leakage preprocessing pipeline adhering to professional ML engineering practices.
* Benchmarked 4 machine learning models on an imbalanced dataset and justified the selection of calibrated SVM.
* Integrated SHAP explainability to provide physical transparency for decision support.
* Deployed an interactive 8-page Streamlit application and validated the entire workflow with automated pytest suites.

#### 8.2 Future Scope
* Integration of real-time IoT sensor telemetry (piezometers, time-domain reflectometry, microseismic geophones).
* Drone photogrammetry integration for automated 3D slope surface mesh extraction.
* Incorporation of temporal sequence models (LSTMs or Temporal Convolutional Networks) for continuous displacement monitoring.

---

### CHAPTER 9: REFERENCES
1. Hoek, E., & Bray, J. (1981). *Rock Slope Engineering*. Institution of Mining and Metallurgy, London.
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765-4774.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
4. Bishop, A. W. (1955). The use of the slip circle in the stability analysis of slopes. *Géotechnique*, 5(1), 7-17.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
