## KNN Diabetes Predictor

### Overview:
This Jupyter Notebook (`KNN_diabetes_predictor.ipynb`) aims to predict the onset of diabetes using the K-Nearest Neighbors (KNN) algorithm. The dataset used contains information about various health metrics and an outcome variable indicating the presence or absence of diabetes.

### Requirements:
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn (`sklearn`)

### Setup and Installation:
1. Ensure that you have Python installed on your system.
2. Install the necessary libraries using pip:
   ```
   pip install pandas numpy scikit-learn
   ```

### Data:
The dataset (`diabetes.csv`) contains the following columns:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (0 or 1 indicating no diabetes or diabetes, respectively)

### Data Preprocessing:
1. Replace zeros in specific columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with the mean of the respective columns.
2. Display the cleaned dataset.

### Model Building:
1. Split the dataset into training and testing sets.
2. Perform feature scaling using `StandardScaler`.
3. Determine the optimal value for K (number of neighbors) using the square root of the test set size. Chose 11 as the number of neighbors.
4. Train a KNN classifier with the chosen parameters.

### Evaluation:
1. Predict the outcomes on the test set.
2. Calculate the confusion matrix.
3. Compute the F1-score and accuracy of the model on the test set.

### Results:
- F1-score: 0.6429
- Accuracy: 0.7403
- Confusion Matrix:
  ```
  [[78, 21],
   [19, 36]]
  ```
This indicates that the model correctly predicted 78 true negatives, 36 true positives, 19 false negatives, and 21 false positives.

### Conclusion:
The KNN model achieved an accuracy of approximately 74% in predicting the onset of diabetes. Further optimization and feature engineering might improve the model's performance.

For further details, refer to the notebook (`KNN_diabetes_predictor.ipynb`).
