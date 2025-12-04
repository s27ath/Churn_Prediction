# 📉 Customer Churn Prediction and Analysis

This project focuses on analyzing a telecommunications dataset to identify key factors contributing to customer churn and building machine learning models to predict which customers are likely to leave the service.

## 🌟 Key Project Highlights

* **Comprehensive Exploratory Data Analysis (EDA):** Univariate and Bivariate analysis to understand the distribution of key features (Gender, Tenure, Monthly Charges, Contract Type) and their relationship with the 'Churn' variable.
* **Data Preprocessing Pipeline:** Handled missing values (imputation of TotalCharges), applied **StandardScaler** to numerical features, and performed **OneHotEncoding** on categorical features.
* **Machine Learning Model Comparison:** Trained and evaluated five different classification models for churn prediction.
* **Key Finding:** Identified **Support Vector Classifier (SVC)** as the best-performing model based on accuracy.

## 🛠️ Technologies and Libraries

* **Python:** Primary programming language.
* **Pandas & NumPy:** Data manipulation and analysis.
* **Matplotlib / Seaborn (Implied by plots):** Data visualization (bar plots, box plots, histograms).
* **Scikit-learn (sklearn):**
    * `train_test_split`: Data splitting.
    * `StandardScaler`, `OneHotEncoder`: Data preprocessing.
    * `KNeighborsClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`: Model building.
    * `metrics`: Model evaluation (Accuracy Score, Confusion Matrix).

## 🚀 Analysis and Model Results

### 1. Exploratory Data Analysis (EDA) Insights

* **Contract:** Most customers are on a **Month-to-month** contract, which is generally associated with higher churn risk.
* **Tenure:** Customers who churn tend to have significantly **lower tenure** compared to non-churning customers.
* **Internet Service:** A high percentage of churn comes from customers using **Fiber Optic** service.
* **Monthly Charges:** Churning customers tend to have **higher monthly charges** on average.

### 2. Model Performance

The following models were trained and evaluated on the test set:

| Model | Accuracy Score |
| :--- | :--- |
| **Support Vector Classifier (SVC)** | **~79.13%** |
| **Logistic Regression** | **~78.56%** |
| Random Forest Classifier | ~77.85% |
| K-Nearest Neighbors (KNN) | ~75.96% |
| Decision Tree Classifier | ~71.65% |

### 3. Conclusion

The **Support Vector Classifier (SVC)** provided the highest accuracy in predicting customer churn on this dataset. Further hyperparameter tuning and feature engineering could be applied to improve all models.

## 📂 Repository Contents

* `Churn Analysis.ipynb`: The main Jupyter Notebook containing all the data cleaning, EDA, preprocessing, model training, and evaluation code.

## ⚙️ How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install the dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  **Run the notebook:** Open the `Churn Analysis.ipynb` file in a Jupyter environment (like JupyterLab or Google Colab) and execute the cells sequentially.
