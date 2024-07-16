# credit_card_fraud_detection

This project aims to create a robust machine learning model for detecting fraudulent credit card transactions. The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle, which includes 284,807 transactions with 492 cases of fraud. Given the highly imbalanced nature of the dataset, special techniques and careful preprocessing were necessary to build an effective model.

Key Components of the Project:
Data Preprocessing:

Handling Missing Values: Ensured that any missing values in the dataset were appropriately dealt with to maintain data integrity.
Feature Scaling: Applied feature scaling techniques to normalize the data, which is crucial for the performance of many machine learning algorithms.
Addressing Class Imbalance: Utilized the Synthetic Minority Over-sampling Technique (SMOTE) to handle the significant class imbalance, increasing the representation of the minority class (fraudulent transactions) to improve model performance.
Exploratory Data Analysis (EDA):

Conducted thorough EDA to understand the underlying patterns and distributions in the data. This included visualizing the distribution of fraudulent and non-fraudulent transactions, correlation analysis, and identifying key features that contribute to fraud detection.
Used visualization tools like Matplotlib and Seaborn to create insightful graphs and plots that aided in understanding the data better.
Model Training and Evaluation:

Logistic Regression: Implemented logistic regression as a baseline model to understand the basic relationships within the data.
Random Forest: Applied a random forest classifier to capture non-linear relationships and interactions between features.
Gradient Boosting: Used gradient boosting techniques to improve prediction accuracy by combining multiple weak learners.
Evaluated model performance using various metrics such as precision, recall, F1-score, and ROC-AUC to ensure a balanced assessment of the model’s ability to detect fraud while minimizing false positives.
Hyperparameter Tuning:

Performed extensive hyperparameter tuning using Grid Search and Random Search to find the optimal settings for each model. This process involved testing various combinations of parameters to enhance the model’s predictive performance.
Applied cross-validation techniques to ensure that the model generalizes well to unseen data.
Visualization and Deployment:

Developed an interactive web application using Streamlit to allow real-time prediction and visualization of fraudulent transactions. The app provides a user-friendly interface for inputting transaction details and obtaining fraud predictions.
Integrated various visual elements to display transaction data and prediction results, making the application both informative and accessible to users.
Tools and Technologies Used:
Jupyter Notebook: For interactive data analysis, model development, and documentation.
Spyder IDE: For script writing, debugging, and testing different approaches.
Streamlit: For building and deploying an interactive web application.
Pandas: For data manipulation and preprocessing.
Scikit-learn: For implementing and evaluating machine learning models.
Matplotlib/Seaborn: For creating visualizations during EDA and presenting model results.
Imbalanced-learn: For handling class imbalance issues using techniques like SMOTE.
