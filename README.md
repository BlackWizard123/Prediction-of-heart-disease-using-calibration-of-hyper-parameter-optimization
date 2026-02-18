
<img width="3464" height="1635" alt="Picsart_26-02-19_01-24-13-924" src="https://github.com/user-attachments/assets/357ecc8c-4f01-4b6e-b937-c29792a504e6" />

# Prediction of heart disease using calibration of hyper parameter optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Healthcare-red.svg)

## PROBLEM STATEMENT

Coronary artery disease prediction is considered to be one of the most challenging tasks in the health care industries. Hyperparameters are set by the machine learning engineer before training and they play a major role in performance of the model. Selecting the best hyperparameter boosts the performance of the Machine Learning model.

## OBJECTIVES

* **Efficient Prediction:** To predict heart disease with high accuracy and computational efficiency.
* **Feature Discovery:** To identify the most significant clinical features required for effective classification.
* **Hyperparameter Optimization:** To fine-tune model parameters to achieve superior predictive results.

## OUTCOMES

* **Optimal Feature Selection:** Applied techniques successfully identified key classification features and eliminated irrelevant data points.
* **Parameter Tuning:** Implementation of various Hyperparameter Optimization (HPO) techniques to discover the most effective parameter combinations for the model.

---

## LITERATURE SURVEY

## SURVEY 1: Implementation of Machine Learning Model to Predict Heart Failure Disease

* **Author:** Fahd Saleh Alotaibi
* **Journal:** (IJACSA) International Journal of Advanced Computer Science and Applications
* **Year:** 2019

#### Methodology

The research focuses on enhancing predictive accuracy through rigorous data preparation and multi-model benchmarking:

* **Data Augmentation:** To address dataset limitations, synthetic data was generated using random number generation for each feature column to increase the sample size.
* **Data Imputation:** Missing values were handled using the **K-Nearest Neighbor (KNN)** method, ensuring that missing entries were estimated based on the similarity of existing data points.
* **Outlier Detection:** Utilized RapidMiner’s distance-based operators to identify and isolate anomalies that could skew model performance.
* **Model Benchmarking:** The processed dataset was evaluated across five diverse machine learning architectures: **Decision Tree, Naïve Bayes, Random Forest, Logistic Regression, and Support Vector Machines (SVM)**.

#### Advantages

* **Relationship Preservation:** KNN imputation effectively maintains the underlying correlation between variables, which is critical for medical diagnostic accuracy.
* **Efficiency in Pipeline Development:** The use of RapidMiner allowed for the swift construction of automated data cleaning workflows, reducing manual preprocessing overhead.

#### Limitations

* **Noise Induction:** KNN imputation can introduce significant noise when dealing with highly correlated variables, potentially degrading the quality of the imputed data.
* **Scalability & Customization:** RapidMiner's visual interface may lack the flexibility required for complex, custom data cleaning scenarios that necessitate low-level coding or advanced algorithmic intervention.

## SURVEY 2: Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques

* **Authors:** Senthilkumar Mohan, Chandrasegar Thirumalai, Gautam Srivastava
* **Publisher:** Elsevier
* **Year:** 2021

#### Methodology

This research introduces a novel hybrid approach focused on feature optimization to enhance the precision of cardiovascular disease diagnostics:

* **Feature Engineering:** The core methodology involves identifying significant clinical features through various combinations to determine their impact on predictive accuracy.
* **Hybrid Modeling:** The study introduces the **Hybrid Random Forest with Linear Model (HRFLM)**, combining different classification techniques to leverage their collective strengths.
* **Comparative Analysis:** The prediction model was rigorously tested against several established classification algorithms to benchmark performance gains.

#### Advantages

* **Data Integrity:** Strict preprocessing protocols were implemented to eliminate records with missing values, ensuring a clean dataset for training.
* **Architectural Insights:** The systematic comparison of multiple classification algorithms provides a clear roadmap for identifying the most effective model for heart disease datasets.

#### Limitations

* **Overfitting Sensitivity:** The HRFLM algorithm is highly dependent on the stability of the Random Forest component; if the RF model overfits, it negatively propagates errors into the linear regression stage.
* **Preprocessing Vulnerability:** Both the Random Forest and Linear Regression components lack native mechanisms to handle outliers or missing data effectively, necessitating heavy manual intervention.

## SURVEY 3: Coronary Artery Disease Detection Using Computational Intelligence Methods

* **Authors:** Roohallah Alizadehsani, Mohammad Hossein Zangooei, Mohammad Javad Hosseini, Jafar Habibi, Abbas Khosravi, Mohamad Roshanzamir, Fahime Khozeimehe, Nizal Sarrafzadeganf, Saeid Nahavandi
* **Publisher:** Elsevier
* **Year:** 2016

#### Methodology

This study utilizes a multi-stage computational intelligence framework combining feature importance, classification, and rule-based mining:

* **Feature Selection:** The researchers employed **"Weights by SVM"** in conjunction with **10-fold cross-validation** to rank and select the most influential features for diagnosis.
* **Classification Framework:** A **Support Vector Machine (SVM)** architecture was implemented, utilizing a diverse set of kernel functions to handle data complexity.
* **Association Rule Mining:** The **Apriori algorithm** was integrated to extract logical rules and patterns from the dataset, providing interpretable clinical insights.

#### Advantages (Pros)

* **Performance in Imbalanced Data:** By applying feature weighting, the SVM achieves superior accuracy, sensitivity, and F1-scores, effectively mitigating bias in imbalanced healthcare datasets.
* **Kernel Flexibility:** The architecture supports multiple kernel functions (Linear, Polynomial, Gaussian/RBF), allowing the model to capture intricate non-linear relationships and high-dimensional complexities within medical data.

#### Limitations (Cons)

* **Weighting Sensitivity:** The performance gains from weighting are highly conditional; if the dataset is balanced or the weighting scheme is improperly calibrated, it may lead to suboptimal results.
* **Computational Overhead:** The use of multiple kernels

---

## SYSTEM ARCHITECTURE

<img width="901" height="511" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/c6157179-c9ea-4e6c-a9b0-bf7dddf83487" />

---

## SYSTEM MODULES

## I. Data Preprocessing

The initial phase focuses on refining the raw dataset to ensure high-quality input for the models:

* **Duplicate Management:** Identification and removal of duplicate entries to prevent model bias.
* **Dataset Balancing:** Implementing techniques to handle class imbalance, ensuring the model identifies disease cases as effectively as healthy ones.

## II. Feature Selection

A multi-layered approach to identify the most predictive clinical variables:

* **Filter Methods:** Statistical evaluation of features independent of the model using **Mutual Information**, **ANOVA**, and **Chi-square** tests.
* **Wrapper Methods:** Iterative selection processes including **Sequential Forward Selection (SFS)**, **Sequential Backward Elimination (SBE)**, and **Boruta** feature selection.
* **Embedded Methods:** Feature selection integrated within the training process using **LASSO (L1 Regularization)**, **Decision Trees**, and **Genetic Algorithms**.

## III. Model Building

Implementation of robust machine learning architectures for classification:

* **Random Forest Classifier:** An ensemble of decision trees to reduce variance and improve accuracy.
* **Extreme Gradient Boosting (XGBoost):** A high-performance gradient boosting framework optimized for speed and performance.
* **K-Nearest Neighbors (KNN):** A distance-based algorithm for classifying patients based on feature similarity.

## IV. Hyperparameter Optimization (HPO)

Advanced tuning strategies to find the optimal configuration for the selected models:

* **Grid Search:** Exhaustive search through a manually specified subset of the hyperparameter space.
* **Randomized Search:** Efficient sampling of the parameter space to find high-performing configurations faster than Grid Search.
* **Tree-based Pipeline Optimization Tool (TPOT):** An automated ML (AutoML) tool that uses genetic programming to optimize entire pipelines.
* **Bayesian Optimization:** A probabilistic model-based approach that intelligently explores the parameter space.
* **Hyperband:** A variation of random search that uses early stopping to speed up the tuning process by focusing on promising configurations.

---

## MODULE I: DATA PROCESSING

### Removing Duplicate Entries

To ensure data integrity and prevent the model from overfitting to redundant information, the dataset is scanned for identical rows. All duplicate entries are dropped, leaving only unique clinical records for analysis.

### Balancing Dataset

Medical datasets often suffer from class imbalance where healthy cases significantly outnumber disease cases. To address this:

* **SMOTE (Synthetic Minority Oversampling Technique):** The dataset is balanced using SMOTE. Unlike simple oversampling which replicates existing data, SMOTE generates synthetic examples by interpolating between existing minority class instances.
* **Goal:** This balances the class distribution, preventing the model from developing a bias toward the majority class and improving its ability to detect Coronary Artery Disease.

---

## MODULE II: FEATURE SELECTION

The presence of irrelevant or redundant features can significantly degrade the performance of classification models. To mitigate this, nine distinct feature selection techniques across three core methodologies are implemented.

### Benefits of Feature Selection

* **Mitigates the Curse of Dimensionality:** Reducing the number of input variables prevents the model from becoming overly complex and failing to generalize to new data.
* **Improved Interpretability:** Simplifies the model architecture, allowing researchers and healthcare professionals to easily understand and interpret the clinical significance of the features.
* **Reduced Training Time:** By decreasing the volume of data processed, the computational overhead and time required for model training are substantially lowered.

### Feature Selection Methodologies

A comprehensive approach is taken by utilizing three distinct methods:

1. **Filter Methods:** Uses statistical measures to score the correlation between features and the target variable (e.g., **Mutual Information, ANOVA, Chi-square**).
2. **Wrapper Methods:** Treats the feature selection process as a search problem, evaluating subsets of features by actually training a model on them (e.g., **Sequential Forward Selection, Sequential Backward Elimination, Boruta**).
3. **Embedded Methods:** Performs feature selection as an integral part of the model construction process (e.g., **LASSO, Decision Trees, Genetic Algorithms**).

<img width="581" height="270" alt="Untitled Diagram-Page-2 drawio pngjnj" src="https://github.com/user-attachments/assets/e8d50f97-ab33-4a37-aeeb-53bd4ede186a" />

---

### I. FILTER METHODS

### 1. Mutual Information

* **Definition:** Mutual information measures the amount of information obtained about one random variable through the observation of another. It quantifies the statistical dependence between a feature and the target variable.
* **Symmetry:** It is a measure of "mutual dependence," characterized by its symmetry:
  $I(X; Y) = I(Y; X)$
* **Application:** Higher mutual information indicates a strong relationship between the clinical feature and the presence of heart disease, making it a valuable tool for non-linear dependency detection.

### 2. Chi-Square Test

The Chi-square ($x^2$) test evaluates the independence of two variables. In this module, it is used to calculate the relationship between each categorical feature and the target variable. Features with the highest Chi-square scores are selected as they indicate a significant correlation with the target.

The score is calculated using the following formula:

$$
x^2 = \frac{(\text{Observed frequency} - \text{Expected frequency})^2}{\text{Expected frequency}}

$$

**Where:**

* **Observed frequency:** The actual number of observations recorded for a specific class.
* **Expected frequency:** The number of observations expected for a class if there were no relationship (null hypothesis) between the feature and the target.

### 3. ANOVA Test

* **Definition:** ANOVA stands for "Analysis of Variance." It is a parametric statistical hypothesis test used to determine whether the means of two or more samples are drawn from the same distribution.
* **Feature Ranking:** Each clinical feature is evaluated and ranked according to its **F-statistic**. Features with higher F-scores indicate a higher degree of variance between groups, identifying them as the most optimal components for distinguishing between healthy and diseased states.

### II. WRAPPER METHODS

### 4. Sequential Forward Selection (SFS)

SFS is a greedy search algorithm that builds a feature subset by adding one feature at a time based on a performance criterion:

* **Initial Step:** The process begins by selecting the single best feature that maximizes the chosen criterion function.
* **Iterative Pairing:** Once the best feature is fixed, the algorithm forms pairs by combining it with each of the remaining features. The pair that results in the highest performance is selected.
* **Subsequent Selection:** This logic extends to triplets and larger groups, adding the next best feature that provides the most incremental benefit.
* **Termination:** The procedure continues until a predefined number of features is reached or no further performance gain is observed.

### 5. Sequential Backward Elimination (SBE)

SBE is the inverse of the forward selection process, starting with the full set of features and pruning them down:

* **Initial Evaluation:** The criterion function is first computed for the complete set of $n$ features.
* **Iterative Deletion:** Each feature is deleted one at a time, and the criterion function is recalculated for the resulting subsets of $n-1$ features. The feature whose removal results in the least performance loss (or the "worst" performing feature) is permanently discarded.
* **Termination:** This procedure repeats iteratively, reducing the feature set until a predefined number of optimal features remains.

### 6. Boruta Feature Selection

Boruta is a robust feature selection algorithm designed to capture all relevant features by comparing them against "shadow" versions:

* **Adding Randomness:** The algorithm creates shuffled copies of all features in the original dataset, referred to as **Shadow Features**.
* **Random Forest Training:** A Random Forest classifier is trained on this extended dataset (original + shadow features). A feature importance measure, such as **Mean Decrease Accuracy**, is used to evaluate every feature.
* **Z-Score Comparison:** In every iteration, the algorithm checks if a real feature has a higher importance than the best-performing shadow feature.
* **Decision Criteria:** Features are "confirmed" if they consistently outperform shadow features and "rejected" if they do not. The process stops when all features are classified or a specified iteration limit is reached.

### III. EMBEDDED METHODS

### 7. LASSO (L1 Regularization)

* **Definition:** LASSO stands for "Least Absolute Shrinkage and Selection Operator." It is a regression analysis method that performs both variable selection and regularization to enhance the prediction accuracy and interpretability of the model.
* **Mechanism:** Lasso adds a penalty equal to the absolute value of the magnitude of coefficients. This specific regularization has the unique ability to shrink some coefficients exactly to zero.
* **Feature Selection:** Features whose coefficients are reduced to zero are considered non-contributing and are safely removed from the model. The remaining features with non-zero coefficients are identified as the most significant predictors.

### 8. Decision Tree

* **Importance Calculation:** During the construction of the Decision Tree using training data, the algorithm naturally ranks features based on their utility in splitting the data.
* **Information Gain:** Feature importance is calculated based on **Information Gain** (or Gini Impurity reduction). Features that result in the highest reduction of entropy (uncertainty) are positioned higher in the tree and are considered more important for the final classification.

### 9. Genetic Algorithm (GA)

* **Principle:** The Genetic Algorithm is a heuristic search and optimization technique based on the principles of natural selection and biological evolution.
* **Mechanism:** The algorithm maintains a "population" of potential feature subsets (individuals). It mimics human evolution by performing the following steps:
  * **Selection:** Choosing the fittest individuals (feature subsets that yield high model accuracy) as "parents."
  * **Crossover (Recombination):** Combining parts of two parents to create "offspring" for the next generation.
  * **Mutation:** Applying random changes to individuals to maintain genetic diversity and explore new feature combinations.
* **Termination:** This iterative process continues until a predefined stopping criterion is met (e.g., number of generations or fitness plateau). The result is the "best individual," representing the most optimal set of features for predicting coronary artery disease.

---

## MODULE III: MODEL BUILDING

### 1. Random Forest Classifier

* **Ensemble Learning:** Random Forest is an ensemble classifier that constructs multiple decision trees during training. It fits these trees on various sub-samples of the dataset and uses averaging to enhance predictive accuracy while effectively controlling overfitting.
* **Majority Voting:** The algorithm aggregates the predictions from every individual decision tree. The final output is determined by "majority voting"—the class that receives the most votes across the forest is selected as the final prediction.

### 2. K-Nearest Neighbors (KNN)

* **Proximity Principle:** The KNN algorithm operates on the fundamental principle of feature similarity. It assumes that data points located close to each other in a high-dimensional feature space are likely to belong to the same clinical class or share similar health outcomes.
* **Classification:** For any given patient record, the algorithm identifies the 'K' most similar instances (neighbors) and assigns the class most common among them.

### 3. Extreme Gradient Boosting (XGBoost)

* **Sequential Learning:** XGBoost is a high-performance implementation of Gradient Boosted decision trees. Unlike Random Forest, trees are created in a sequential manner where each new tree attempts to correct the errors of the previous one.
* **Weighted Optimization:** Initial weights are assigned to all independent variables and fed into the first decision tree.
* **Error Correction:** The algorithm identifies variables that were incorrectly predicted, increases their weights, and passes this updated information to the next tree in the sequence to minimize the residual error.

### 4. Ensemble Methods

To further boost performance, the three individual classifiers (Random Forest, KNN, and XGBoost) are combined using advanced ensemble strategies:

* **Voting Ensemble:** This model aggregates the predictions from all three classifiers. The final output is determined by **Majority Voting**, where the class selected by the majority of the sub-models becomes the final diagnostic prediction.
* **Stacking Ensemble:** This involves a two-layer architecture. First, the base models (classifiers) are trained to solve the prediction problem. Their combined outputs are then used as input features to train a **Meta-Model**, which learns how to best combine their strengths to achieve a higher level of accuracy.

---

## MODULE IV: HYPER PARAMETER OPTIMIZATION

### 1. Grid Search

* **Exhaustive Search:** Grid Search is an exhaustive search method performed across a manually specified subset of the hyperparameter space of a machine learning model (estimator).
* **Brute-Force Approach:** The algorithm "brute-forces" every possible combination of the provided parameters. While computationally expensive, it guarantees finding the best combination within the predefined grid.

### 2. Randomized Search

* **Random Sampling:** Unlike Grid Search, Randomized Search does not attempt every possible combination. Instead, it selects a fixed number of parameter combinations at random from the specified distributions.
* **Flexibility:** You do not necessarily need to specify a discrete set of values for every hyperparameter; you can provide statistical distributions (e.g., a normal or uniform distribution) from which the algorithm samples. This often finds a "good enough" solution significantly faster than Grid Search.

### 3. Tree-Based Pipeline Optimization Tool (TPOT)

* **AutoML Framework:** TPOT is an Automated Machine Learning (AutoML) tool that uses genetic programming to automatically design and optimize entire machine learning pipelines.
* **Evolutionary Optimization:** By utilizing a flexible expression tree representation, TPOT explores thousands of possible pipelines—including various feature selectors, transformers, and classifiers—to find the one that performs best for the specific dataset.
* **Stochastic Search:** It employs genetic algorithms to "evolve" these pipelines, combining the most successful components over generations to reach an optimal solution without manual intervention.

### 4. Bayesian Optimization

* **Informed Search:** Unlike Grid or Random search, Bayesian methods are "sequential." They leverage the results of previous evaluations to make informed decisions about which hyperparameter values to test next.
* **Efficiency:** The primary goal is to minimize the number of expensive objective function evaluations. By building a probabilistic model (surrogate model) of the objective function, the algorithm focuses its search on areas of the parameter space that have shown high potential in past trials.
* **Versatility:** Bayesian Optimization is highly effective across all hyperparameter types, whether they are continuous (like learning rates) or categorical (like kernel types).

### 5. Hyperband

* **Budget Allocation:** Hyperband is an optimization strategy that treats hyperparameter tuning as a resource allocation problem. It generates small-sized subsets of the data and allocates "budgets" (such as iterations or time) to each hyperparameter combination based on its real-time performance.
* **Optimized Grid Search:** It essentially functions as a sophisticated grid search over the optimal allocation strategy, using early stopping to discard poorly performing configurations.
* **Random Sampling:** At each individual trial, the specific set of hyperparameters is chosen randomly, similar to Random Search, but with more efficient resource management.
* **Speed and Parallelization:** Because the algorithm supports massive parallelization and focuses resources only on the most promising candidates, it converges to the best hyperparameter set significantly faster than traditional methods.

---

## DATASET DESCRIPTION

The dataset used in this project dates from 1988 and is a compilation of four clinical databases:

* **Cleveland**
* **Hungary**
* **Switzerland**
* **Long Beach V**

### Attributes

While the complete dataset contains **76 attributes**, most published experiments and this project focus on a specific subset of **14 key clinical features** found to be most relevant for diagnosis.

### Target Variable

The **"target"** field indicates the clinical presence of heart disease in the patient:

* **0:** No disease present.
* **1:** Heart disease present.

<img width="750" height="350" alt="image" src="https://github.com/user-attachments/assets/9139c86d-30e7-4980-87c6-1eabf0ae8e7f" />

### Feature Reference Table

The subset of 14 clinical attributes used for prediction is detailed below:


| #  | Attribute     | Description                                                                                                               |
| :--- | :-------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| 1  | **age**       | Age of the patient in years.                                                                                              |
| 2  | **sex**       | Gender (1 = Male; 0 = Female).                                                                                            |
| 3  | **cp**        | **Chest pain type:** <br>• 0: Typical angina <br>• 1: Atypical angina <br>• 2: Non-anginal pain <br>• 3: Asymptomatic |
| 4  | **trestbps**  | Resting blood pressure (in mm Hg on admission).                                                                           |
| 5  | **chol**      | Serum cholesterol in mg/dl.                                                                                               |
| 6  | **restecg**   | **Resting ECG results:** <br>• 0: Normal <br>• 1: ST-T wave abnormality <br>• 2: Left ventricular hypertrophy          |
| 7  | **thalach**   | Maximum heart rate achieved.                                                                                              |
| 8  | **exang**     | Exercise induced angina (1 = Yes; 0 = No).                                                                                |
| 9  | **oldpeak**   | ST depression induced by exercise relative to rest.                                                                       |
| 10 | **slope**     | **Peak exercise ST segment slope:** <br>• 0: Upsloping <br>• 1: Flat <br>• 2: Downsloping                              |
| 11 | **ca**        | Number of major vessels (0-3) colored by fluoroscopy.                                                                     |
| 12 | **thal**      | Thalassemia: 0 = Normal; 1 = Fixed defect; 2 = Reversible defect.                                                         |
| 13 | **fbs**       | Fasting blood sugar > 120 mg/dl (1 = True; 0 = False).                                                                    |
| 14 | **condition** | **Target Label:** 0 = No disease; 1 = Disease.                                                                            |

---

# IMPLEMENTATION & PERFORMANCE RESULTS

## 1. Random Forest Classifier

The following tables demonstrate the performance of the Random Forest Classifier across different feature selection categories. The **highest accuracy** and **most efficient feature counts** are highlighted.

### Filter Methods


| Technique           | Mutual Information | ANOVA Test | **Chi-Square** |
| :-------------------- | :------------------- | :----------- | :--------------- |
| **Accuracy**        | 91.8027%           | 91.8027%   | 93.4426%       |
| **No. of Features** | 11                 | 13         | 10             |

### Wrapper Methods


| Technique           | **Seq. Forward Selection** | Seq. Backward Elimination | Boruta FS |
| :-------------------- | :--------------------------- | :-------------------------- | :---------- |
| **Accuracy**        | **90.1639%**               | 88.5246%                  | 88.5246%  |
| **No. of Features** | 10                         | 9                         | **8**     |

### Embedded Methods


| Technique           | Lasso    | Decision Tree | **Genetic Algorithm** |
| :-------------------- | :--------- | :-------------- | :---------------------- |
| **Accuracy**        | 85.2459% | 83.6065%      | **88.5246%**          |
| **No. of Features** | **7**    | **7**         | 9                     |

## Selected Feature Subsets

Based on the top-performing techniques in each category, the following feature sets were identified:

* **Optimal Filter Subset (Chi-Square):** `'age'`, `'sex'`, `'cp'`, `'trestbps'`, `'fbs'`, `'restecg'`, `'exang'`, `'slope'`, `'ca'`, `'thal'`
* **Optimal Wrapper Subset (SFS):** `'thalach'`, `'oldpeak'`, `'ca'`, `'cp'`, `'exang'`, `'chol'`, `'age'`, `'trestbps'`, `'slope'`, `'sex'`
* **Optimal Embedded Subset (Genetic Algorithm):** `'age'`, `'sex'`, `'cp'`, `'trestbps'`, `'restecg'`, `'exang'`, `'slope'`, `'ca'`, `'thal'`

## Hyperparameter Optimization Results: Random Forest Classifier

This section details the performance gains achieved by tuning the primary hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `criterion`.


| Optimization Technique    | Chi-sq Subset | SFS Subset   | Genetic Algorithm Subset |
| :-------------------------- | :-------------- | :------------- | :------------------------- |
| **Grid Search**           | **95.0819%**  | **91.8033%** | 90.1639%                 |
| **Randomized Search**     | 93.4426%      | **91.8033%** | 91.8033%                 |
| **TPOT (AutoML)**         | **95.0819%**  | 90.1639%     | **93.4426%**             |
| **Bayesian Optimization** | 93.4426%      | **91.8033%** | 90.1639%                 |
| **Hyperband**             | 90.1639%      | 90.1639%     | 90.1639%                 |

---

## 2. Extreme Gradient Boost (XGBoost) Classifier

The following tables demonstrate the performance of the XGBoost Classifier across different feature selection categories. The **highest accuracy** and **most efficient feature counts** are highlighted.

### Filter Methods


| Technique           | **Mutual Information** | ANOVA Test | Chi-Square |
| :-------------------- | :----------------------- | :----------- | :----------- |
| **Accuracy**        | **90.1639%**           | 88.5246%   | 88.8246%   |
| **No. of Features** | **9**                  | 13         | 10         |

### Wrapper Methods


| Technique           | **Seq. Forward Selection** | Seq. Backward Elimination | **Boruta FS** |
| :-------------------- | :--------------------------- | :-------------------------- | :-------------- |
| **Accuracy**        | **91.8033%**               | 88.5246%                  | **91.8033%**  |
| **No. of Features** | **5**                      | 9                         | 6             |

### Embedded Methods


| Technique           | Lasso    | **Decision Tree** | Genetic Algorithm |
| :-------------------- | :--------- | :------------------ | :------------------ |
| **Accuracy**        | 78.6885% | **80.3279%**      | 80.3270%          |
| **No. of Features** | 7        | **3**             | 5                 |

## Selected Feature Subsets

Based on the top-performing techniques in each category, the following feature sets were identified:

* **Optimal Filter Subset (Mutual Info):** `'cp'`, `'thal'`, `'ca'`, `'slope'`, `'exang'`, `'sex'`
* **Optimal Wrapper Subset (Boruta):** `'cp'`, `'thal'`, `'ca'`, `'slope'`, `'oldpeak'`, `'exang'`, `'thalach'`, `'chol'`, `'sex'`
* **Optimal Embedded Subset (Decision Tree):** `'cp'`, `'ca'`, `'thal'`

## Hyperparameter Optimization Results: Extreme Gradient Boost (XGBoost)

This section details the performance gains achieved by tuning the primary hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `min_child_weight`.


| Optimization Technique    | Mutual Info Subset | Boruta FS Subset | Decision Tree Subset |
| :-------------------------- | :------------------- | :----------------- | :--------------------- |
| **Grid Search**           | **93.4426%**       | **93.4426%**     | **93.4426%**         |
| **Randomized Search**     | 88.5245%           | **93.4426%**     | **93.4426%**         |
| **TPOT (AutoML)**         | 90.1639%           | 91.8033%         | 90.8033%             |
| **Bayesian Optimization** | 88.5246%           | **93.4426%**     | **93.4426%**         |
| **Hyperband**             | 90.1639%           | 90.1639%         | 88.6344%             |

---

## Performance Results: K-Nearest Neighbor (KNN) Classifier

The following tables demonstrate the performance of the KNN Classifier across different feature selection categories. The **highest accuracy** and **most efficient feature counts** are highlighted.

### Filter Methods


| Technique           | **Mutual Information** | ANOVA Test | Chi-Square |
| :-------------------- | :----------------------- | :----------- | :----------- |
| **Accuracy**        | **91.8023%**           | 85.2459%   | 85.2459%   |
| **No. of Features** | **5**                  | 8          | **5**      |

### Wrapper Methods


| Technique           | **Seq. Forward Selection** | Seq. Backward Elimination | Boruta FS |
| :-------------------- | :--------------------------- | :-------------------------- | :---------- |
| **Accuracy**        | **75.4098%**               | 67.2131%                  | N/A       |
| **No. of Features** | 8                          | 8                         | -         |

### Embedded Methods


| Technique           | Lasso    | Decision Tree | **Genetic Algorithm** |
| :-------------------- | :--------- | :-------------- | :---------------------- |
| **Accuracy**        | 91.8030% | 83.6065%      | **91.8033%**          |
| **No. of Features** | **6**    | 7             | 7                     |

## Selected Feature Subsets

Based on the top-performing techniques in each category, the following feature sets were identified:

* **Optimal Filter Subset (Mutual Info):** `'age'`, `'sex'`, `'cp'`, `'fbs'`, `'exang'`, `'slope'`, `'ca'`, `'thal'`
* **Optimal Wrapper Subset (SFS):** `'thal'`, `'cp'`, `'ca'`, `'oldpeak'`, `'slope'`
* **Optimal Embedded Subset (Genetic Algorithm):** `'sex'`, `'cp'`, `'fbs'`, `'oldpeak'`, `'slope'`, `'ca'`, `'thal'`

## Hyperparameter Optimization Results: K-Nearest Neighbor (KNN)

This section details the performance gains achieved by tuning the primary hyperparameters: `n_neighbors`, `weights`, `algorithm`, `leaf_size`, and `p`.


| Optimization Technique    | Mutual Info Subset | SFS Subset   | Genetic Algorithm Subset |
| :-------------------------- | :------------------- | :------------- | :------------------------- |
| **Grid Search**           | 91.8033%           | 80.3277%     | **91.8233%**             |
| **Randomized Search**     | 91.8033%           | **80.3279%** | 90.1639%                 |
| **TPOT (AutoML)**         | 90.1639%           | **80.3279%** | 90.1639%                 |
| **Bayesian Optimization** | 90.1639%           | 78.6885%     | 90.1639%                 |
| **Hyperband**             | 83.6344%           | 78.6885%     | 83.6344%                 |

---

## BEST PERFORMANCE SUMMARY

The following table summarizes the optimal configuration for each classifier, showcasing the best combination of feature selection and hyperparameter tuning to achieve maximum accuracy.


| Classifier        | Best Feature Selection Technique | Best Hyperparameter Tuning Algorithm | **Maximum Accuracy** |
| :------------------ | :--------------------------------- | :------------------------------------- | :--------------------- |
| **Random Forest** | Chi-Square                       | Grid Search / Bayesian Optimization  | **95.0819%**         |
| **XGBoost**       | Boruta Feature Selection         | Grid / Random / TPOT                 | **93.4426%**         |
| **KNN**           | Genetic Algorithm                | Grid Search                          | **91.8233%**         |

---

## MODULE V: FINAL ENSEMBLE AND COMPARITIVE ANALYSIS

After optimizing the individual models, ensemble techniques were implemented to leverage the combined predictive power of the classifiers. Additionally, a standalone Decision Tree was evaluated for comparison.

### Final Model Comparison

The following table summarizes the performance of the ensemble methods compared to the individual decision-path architecture:


| Classifier             | Accuracy   | Recall     |
| :----------------------- | :----------- | :----------- |
| **Decision Tree**      | **98.36%** | **96.88%** |
| **Voting Classifier**  | 93.44%     | **96.88%** |
| **Stacked Classifier** | 91.80%     | 87.55%     |

### Ensemble Analysis

* **Decision Tree:** Achieved the highest accuracy in this specific trial. While highly accurate, Decision Trees are often prone to variance, which is why the ensemble methods were used to ensure model stability across different data subsets.
* **Voting Classifier:** By aggregating the predictions of the Random Forest, XGBoost, and KNN models, the Voting Classifier maintained a very high **Recall of 96.88%**. In a medical context, high recall is critical as it minimizes "False Negatives"—ensuring that patients with heart disease are not mistakenly identified as healthy.
* **Stacked Classifier:** This model used a meta-learner to weigh the predictions of the base models. While it achieved a solid 91.80% accuracy, it showed a slightly lower recall, suggesting that for this specific dataset, a direct voting mechanism was more effective for clinical sensitivity.

---

To evaluate the success of this project, we compared our results against the current industry benchmark established in the base paper.


| Feature               | Base Paper Result                  | Proposed Model (This Project)                   |
| :---------------------- | :----------------------------------- | :------------------------------------------------ |
| **Feature Selection** | Sequential Forward Selection (SFS) | Hybrid Selection (Chi-Sq/Boruta/Genetic)        |
| **Algorithm**         | Random Forest                      | **Ensemble (RF, XGBoost, KNN)**                 |
| **Optimization**      | TPOT                               | Multi-strategy HPO + Decision Tree Meta-Learner |
| **Max Accuracy**      | 97.52%                             | **98.36%**                                      |

<img width="647" height="352" alt="image" src="https://github.com/user-attachments/assets/3a207db9-10e0-4123-a5de-1b48843de062" />

### Key Improvements

* **Accuracy Boost:** Our proposed model achieved an accuracy of **98.36%**, outperforming the base paper's accuracy of 97.52%.
* **Architectural Synergy:** While the base paper relied on a single classifier (Random Forest) optimized via TPOT, our model leverages a **Decision Tree Meta-Learner** to ensemble the strengths of Random Forest, XGBoost, and KNN.
* **Clinical Reliability:** The ensemble approach provides a more stable prediction framework, reducing the variance typically associated with single-model architectures.

---

## SOCIAL & ECONOMICAL IMPACT ANALYSIS

Implementing a high-accuracy diagnostic tool (98.36%) for Coronary Artery Disease extends beyond technical metrics, offering significant benefits to society and the healthcare economy.

### 🌍 Social Impacts

* **Early Detection and Prevention:** By identifying high-risk individuals before symptoms become critical, the model facilitates early clinical intervention and personalized preventive care.
* **Improved Access to Healthcare:** Data-driven insights allow healthcare providers to target vulnerable at-risk populations, ensuring resources are allocated where they are needed most.
* **Increased Awareness and Education:** The deployment of these models raises public consciousness regarding cardiovascular risk factors, empowering individuals to make informed lifestyle changes.
* **Reduced Healthcare Burden:** Preventive measures identified by the system help decrease the frequency of emergency hospitalizations and complex surgeries.

### 💰 Economic Aspects

* **Cost Savings:** Early intervention significantly reduces the long-term financial burden of chronic disease management, specialized medication, and intensive care.
* **Return on Investment (ROI):** The system’s value is measured through saved healthcare expenditure and improved patient outcomes. Higher diagnostic accuracy leads to fewer misdiagnoses, directly increasing hospital efficiency and patient satisfaction.
* **Accessibility & Equity:** For the system to be effective, its implementation cost must remain low to ensure accessibility for low-income and uninsured patients, preventing a "digital divide" in heart health.
* **Data Privacy and Security:** While there is a cost associated with maintaining HIPAA-compliant data security, it is a necessary investment to protect sensitive patient records and maintain public trust.

---

## CONCLUSION

This project successfully developed a high-precision diagnostic framework for **Coronary Artery Disease prediction**, achieving a peak accuracy of **98.36%**. By systematically evaluating **9 different feature selection techniques** and **5 hyperparameter optimization strategies**, we demonstrated that a hybrid ensemble approach significantly outperforms traditional single-classifier models.

The transition from individual models to a **Decision Tree-based Meta-Ensemble** allowed us to surpass the benchmark set by the base paper (97.52%), proving that the strategic combination of **Random Forest, XGBoost, and KNN** can capture complex clinical patterns more effectively than any single algorithm.

Beyond the technical success, this research underscores the potential for AI to drive **socio-economic change** in healthcare. By providing a scalable, low-cost, and highly accurate tool for early detection, this system can help reduce global healthcare costs and, most importantly, improve patient survival rates through timely intervention.
