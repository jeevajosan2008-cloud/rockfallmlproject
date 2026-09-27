# ACADEMIC PRESENTATION SLIDE DECK (12 SLIDES)

## Explainable AI-Based Rockfall Risk Prediction & Early Warning System for Open-Pit Mines

*Use this deck to prepare your PowerPoint presentation or convert directly to slides using Marp / Reveal.js.*

---

### SLIDE 1: Title Slide
* **Title:** Explainable AI-Based Rockfall Risk Prediction and Early Warning System for Open-Pit Mines
* **Subtitle:** An Educational Geotechnical Decision-Support Prototype
* **Presenter:** [Your Name] — 2nd-Year B.E. Computer Science and Engineering
* **Department:** Department of Computer Science & Engineering
* **Academic Year:** 2025–2026

> **Speaker Notes:**  
> "Good morning respected evaluators and panel members. Today, I am presenting our project: 'Explainable AI-Based Rockfall Risk Prediction and Early Warning System for Open-Pit Mines'. This project bridges geotechnical engineering with machine learning and explainable AI to provide a rapid, transparent screening tool for slope stability hazards."

---

### SLIDE 2: Problem Statement & Motivation
* **The Problem:** In surface open-pit mining, bench slope failures and rockfalls represent catastrophic risks to personnel, equipment (shovels, dumpers), and haul road operations.
* **Limitations of Traditional Methods:**
  * Limit-Equilibrium Methods (LEM) such as Bishop or Janbu require intensive geometric modeling and iterative computational solving.
  * Not scalable for rapid, interactive multi-bench screening across dynamic mining environments.
* **Proposed Solution:** A data-driven machine learning classification pipeline combined with Explainable AI (SHAP) for transparent, physics-aligned hazard screening.

> **Speaker Notes:**  
> "In mining engineering, slope geometry is continuously excavated. Traditional numerical simulation tools take hours to model slip surfaces. Our objective was to build a machine-learning prototype capable of evaluating slope stability in milliseconds while explaining the physical parameters contributing to each prediction."

---

### SLIDE 3: Geotechnical Background & Target Formulation
* **Ground-Truth Parameter:** Factor of Safety ($FS$)
  $$FS = \frac{\text{Resisting Shear Strength } (\tau_f)}{\text{Driving Shear Stress } (\tau)}$$
* **Limit Equilibrium Criterion:**
  * $FS \ge 1.0 \implies \text{STABLE}$ (Resisting strength exceeds driving shear)
  * $FS < 1.0 \implies \text{RISK / UNSTABLE}$ (Shear failure occurs)
* **Binary Target Formulation:**
  $$\text{Risk Target} = (FS < 1.0) \implies \begin{cases} 0 & \text{STABLE } (97.05\%) \\ 1 & \text{RISK / UNSTABLE } (2.95\%) \end{cases}$$
* **Class Imbalance:** 33:1 ratio (necessitates cost-sensitive weighting and stratified sampling).

> **Speaker Notes:**  
> "The target variable is directly derived from geotechnical mechanics. A Factor of Safety below 1.0 defines an unstable slope. In our 10,000-sample dataset, 295 samples represent critical failure states, creating an imbalanced distribution that guided our metric selection."

---

### SLIDE 4: Dataset Exploration & Feature Schema
* **Total Dataset Size:** 10,000 records, 9 columns, 0 missing values, 0 duplicates.
* **Continuous Geotechnical Features (6):**
  * `Unit Weight (kN/m³)`: Bulk rock/soil mass density ($15.0 - 25.0\,\text{kN/m}^3$)
  * `Cohesion (kPa)`: Inherent material shear strength ($5.0 - 50.0\,\text{kPa}$)
  * `Internal Friction Angle (°)`: Shearing resistance angle ($20.0^\circ - 45.0^\circ$)
  * `Slope Angle (°)`: Face inclination ($10.0^\circ - 60.0^\circ$)
  * `Slope Height (m)`: Vertical bench height ($5.0 - 50.0\,\text{m}$)
  * `Pore Water Pressure Ratio (r_u)`: Groundwater pore pressure ratio ($0.0 - 1.0$)
* **Categorical Mitigation Feature (1):**
  * `Reinforcement Type`: Drainage, Geosynthetics, Retaining Wall, Soil Nailing.

> **Speaker Notes:**  
> "The dataset captures key Mohr-Coulomb shear strength parameters—cohesion and internal friction—alongside driving forces like slope angle, height, and groundwater pore water pressure, plus physical engineering support measures."

---

### SLIDE 5: Zero-Leakage Preprocessing Pipeline
* **Data Leakage Prevention:**
  * Dataset partitioned **first** into 80% Train (8,000 samples) and 20% Test (2,000 samples) using **Stratified Sampling**.
  * Preprocessing statistics computed **strictly on the training partition**.
* **Scikit-Learn ColumnTransformer Architecture:**
  * `StandardScaler`: Applied to 6 continuous features ($z = \frac{x - \mu}{\sigma}$).
  * `OneHotEncoder`: Applied to `Reinforcement Type` with fixed categories.
* **Collinear Feature Exclusion:** `Reinforcement Numeric` excluded due to 1-to-1 redundancy with `Reinforcement Type`.

> **Speaker Notes:**  
> "To maintain strict ML engineering integrity, we ensured zero data leakage. All scaling means and standard deviations were learned solely from the training data. The hold-out test set was untouched until evaluation."

---

### SLIDE 6: Machine Learning Models Benchmark
* We trained and evaluated 4 distinct classification architectures:
  1. **Logistic Regression:** Linear baseline with balanced class weights.
  2. **Support Vector Machine (SVM):** Non-linear RBF kernel with Platt scaling calibration.
  3. **Random Forest:** Ensemble of 150 bagged decision trees with cost-sensitive leaf splitting.
  4. **XGBoost:** Gradient-boosted decision trees with `scale_pos_weight = 32.89`.
* **Reproducibility:** Fixed random state ($42$) and identical cross-validation folds.

> **Speaker Notes:**  
> "We evaluated four diverse model families ranging from linear baselines to non-linear kernel methods and ensemble tree boosting, tuning each for class-imbalance sensitivity."

---

### SLIDE 7: Empirical Results & Evaluation Metrics
* **Hold-Out Test Set Performance (2,000 Samples):**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (Selected)** | **0.9945** | **0.9444** | **0.8644** | **0.9027** | **0.9983** |
| **XGBoost** | 0.9915 | 0.8500 | 0.8644 | 0.8571 | 0.9979 |
| **Random Forest** | 0.9900 | 0.8545 | 0.7966 | 0.8246 | 0.9942 |
| **Logistic Regression** | 0.9775 | 0.5673 | 1.0000 | 0.7239 | 0.9999 |

* **Confusion Matrix (SVM):**
  * True Negatives: $1,938 \quad|\quad$ False Positives: $3$
  * False Negatives: $8 \quad|\quad$ True Positives: $51$

> **Speaker Notes:**  
> "Looking at the empirical results, Logistic Regression had 100% recall but 45 false alarms, reducing precision to 56%. SVM achieved the highest F1-score of 0.9027, successfully detecting 86.44% of hazards with only 3 false alarms out of 2,000 samples."

---

### SLIDE 8: Why Support Vector Machine (SVM) was Selected
* **The Hazard Screening Trade-Off:**
  * **False Negatives:** In mining safety, a missed hazard could result in an unpredicted slope collapse.
  * **False Positives:** Excessive false alarms lead to alarm fatigue and unnecessary bench evacuations.
* **Selection Justification:**
  * Highest F1-Score (**$0.9027$**) among all candidates.
  * Precision of **$94.44\%$** ensures high trust in triggered alerts.
  * Recall of **$86.44\%$** matches XGBoost in identifying unstable slopes.
  * Calibrated Platt scaling delivers statistically reliable posterior probabilities.

> **Speaker Notes:**  
> "We selected SVM with calibrated RBF kernel because it provides the best operational balance: high sensitivity to true slope hazards without overwhelming mining engineers with false alarms."

---

### SLIDE 9: Explainable AI — Global & Local SHAP
* **Why XAI?** Geotechnical engineers cannot trust an unexplainable 'black box'.
* **Global Insights (What drives stability overall?):**
  * `Cohesion` and `Internal Friction Angle` are the strongest stabilizing factors.
  * `Slope Angle` and `Pore Water Pressure Ratio` are the primary destabilizing drivers.
* **Local Waterfall Insights (Why did this slope fail?):**
  * Decomposes individual predictions into additive SHAP values ($\phi_i$).
  * Shows exact parameters pushing the slope toward safety (green) or failure (red).

> **Speaker Notes:**  
> "SHAP uses cooperative game theory to explain model decisions. It mathematically proves that our model aligns with physical mechanics: high cohesion stabilizes slopes, while steep angles and elevated water pressure trigger failure."

---

### SLIDE 10: Streamlit Application Architecture
* **Modular Interactive Dashboard (8 Pages):**
  1. **Dashboard:** Architecture diagram, KPIs, and data alerts.
  2. **Risk Prediction:** Dynamic inputs, large 🟢/🔴 status card, probability gauge.
  3. **Explainable AI:** Interactive global and local SHAP charts.
  4. **Model Comparison:** Real metrics, confusion matrix heatmaps, ROC curves.
  5. **Risk Analysis:** Factor of Safety distributions, correlations, boxplots.
  6. **What-If Analysis:** Side-by-side scenario simulation with $\Delta\%$ tracking.
  7. **Dataset Explorer:** Filterable raw data table and descriptive statistics.
  8. **Project Information:** Technical documentation and academic disclaimers.

> **Speaker Notes:**  
> "We deployed the complete system as an 8-page interactive Streamlit web application. It allows geotechnical staff to test scenarios, view visual early-warning cards, and inspect SHAP attributions in real time."

---

### SLIDE 11: What-If Sensitivity Analysis Demo
* **Scenario A (Unstable Baseline):**
  * Angle: $55^\circ$, Height: $45\,\text{m}$, Cohesion: $10\,\text{kPa}$, Pore Water Ratio: $0.80$.
  * Result: 🔴 **RISK / UNSTABLE** (Predicted Probability: $> 95\%$).
* **Scenario B (Engineered Drainage & Bench Modification):**
  * Angle reduced to $35^\circ$, Pore Water Ratio reduced to $0.15$ via horizontal drains.
  * Result: 🟢 **STABLE** (Predicted Probability: $< 5\%$).
* **Outcome:** System quantifies the **Probability Delta ($\Delta\% = -90.5\%$)**, demonstrating how engineering interventions restore stability.

> **Speaker Notes:**  
> "Our What-If feature empowers engineers to simulate mitigation measures. In this example, installing drainage and re-benching shifts the predicted failure probability from 95% down to under 5%."

---

### SLIDE 12: Conclusion & Limitations
* **Key Achievements:**
  * End-to-end reproducible ML pipeline with zero data leakage.
  * Rigorous empirical evaluation across 4 algorithms.
  * Integrated Explainable AI (SHAP) for transparent decision support.
  * Automated testing suite ($100\%$ pytest pass rate).
* **Academic Limitations:**
  * Uses slope stability simulation data as a proxy; not trained on historical disaster records or live IoT sensor streams.
  * Decision-support prototype, not a certified operational mine safety system.
* **Thank You! Questions & Discussion**

> **Speaker Notes:**  
> "In conclusion, we have built a complete, explainable decision-support prototype for slope stability risk screening. Thank you for your time, and I am now ready to take your questions."
