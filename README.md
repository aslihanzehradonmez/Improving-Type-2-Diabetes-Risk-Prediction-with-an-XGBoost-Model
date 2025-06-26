# Type 2 Diabetes Risk Prediction with XGBoost 🩺

This project develops a high-performance machine learning model to predict the onset of Type 2 diabetes using the PIMA Indians Diabetes dataset. It utilizes a sophisticated XGBoost classifier integrated into a scikit-learn Pipeline, which handles advanced data imputation and feature scaling. The model is rigorously optimized through GridSearchCV to achieve the highest possible predictive accuracy, demonstrating a professional, end-to-end approach to solving a real-world classification problem.

## ✨ Key Features

  * 🧠 **Advanced XGBoost Model:** Implements a gradient-boosted decision tree model known for its performance and accuracy.
  * 📊 **Pima Indians Diabetes Dataset:** Utilizes the classic binary classification dataset to predict the onset of diabetes.
  * 🛠️ **Feature Engineering:** Creates new predictive features from existing data (e.g., `Glucose_x_BMI`, `BMICategory`, `GlucoseCategory`) to enhance model performance.
  * 💪 **Robust Preprocessing Pipeline:** Integrates an `IterativeImputer` to handle missing values and a `StandardScaler` for feature scaling within a Scikit-learn `Pipeline`.
  * ⚖️ **Handling Class Imbalance:** Addresses the uneven class distribution in the dataset by using the `scale_pos_weight` parameter to give more importance to the minority class (diabetic cases).
  * ⚙️ **Hyperparameter Tuning:** Employs `GridSearchCV` with `StratifiedKFold` cross-validation to find the optimal hyperparameters for the XGBoost model, with a focus on maximizing recall.
  * 🎯 **Threshold Optimization:** Manually adjusts the prediction threshold to achieve a better balance between precision and recall, prioritizing the correct identification of at-risk individuals.

## 🛠️ Tech Stack

  * 🐍 Python 3
  * 🐼 Pandas
  * 🔢 NumPy
  * 🤖 Scikit-learn
  * 🚀 XGBoost
  * 📊 Matplotlib & Seaborn

## ▶️ Quick Start

1.  **Clone the Repository:**

    You can clone the repository to your local machine by running the following command in your terminal:

    ```bash
    git clone https://github.com/aslihanzehradonmez/Improving-Type-2-Diabetes-Risk-Prediction-with-an-XGBoost-Model.git
    ```

    After cloning, navigate into the project directory:

    ```bash
    cd Improving-Type-2-Diabetes-Risk-Prediction-with-an-XGBoost-Model
    ```

2.  **Install dependencies:**

    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyterlab
    ```

3.  **Run the Jupyter Notebook:**

    Launch Jupyter and run the notebook `Improving Type 2 Diabetes Risk Prediction with an XGBoost Model.ipynb`. This single notebook handles everything from data loading and preprocessing to model training, evaluation, and visualization.

## 🏆 Outcome

This project successfully develops an XGBoost classifier optimized for a medical diagnosis task. By implementing feature engineering, handling class imbalance, and fine-tuning the prediction threshold, the model achieves a high **recall of 0.87** for the diabetic class and an overall **accuracy of 77%** on the test set. The primary outcome is a well-documented workflow demonstrating how to build a robust and practical classification model for healthcare, where correctly identifying positive cases (high recall) is often more critical than overall accuracy. The final confusion matrix visually confirms the model's effectiveness in minimizing false negatives.