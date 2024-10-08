
# Machine Learning Models on Heart Disease Dataset
This project explores four popular machine learning models: Decision Tree, AdaBoost, Random Forest, and Bagging to predict heart disease using a public dataset (heart.csv). Below is a summary of each model, its implementation, and evaluation results.

## 1. Decision Tree Classifier
A supervised learning method that splits data into smaller subsets using decision rules. It handles both numerical and categorical data.
<br />Parameters: criterion='gini', max_depth=4, random_state=0
<br />Results: 0.75

![image](https://github.com/user-attachments/assets/5f619b56-a17e-4e24-98bb-dde2446d9834)


## 2. AdaBoost Classifier
Boosting algorithm that turns weak learners into strong ones by weighting them.
<br />Parameters: n_estimators=100, learning_rate=1.0
<br />Results: AUC Score: 0.81

## 3.Random Forest Classifier
An ensemble method that builds multiple decision trees and merges their predictions to reduce overfitting.
<br />Parameters: n_estimators=19, max_depth=7, random_state=3
<br />Results: AUC Score: 0.77

![image](https://github.com/user-attachments/assets/27a4b032-b829-4baf-9102-22611e7c61f6)


## 4. Bagging Classifier
A technique that trains multiple base learners on random subsets of the training data and aggregates their predictions.
<br />Parameters: n_estimators=20, max_samples=0.8, max_features=10
<br />Results: AUC Score: 0.80

![image](https://github.com/user-attachments/assets/c83c8555-7363-4d5f-aef8-f8279215f939)


## 5. Conclusion
The AdaBoost model performed best with an AUC score of 0.81, making it the most effective model for this dataset.

