# 🧠 Deep Learning Challenge: Alphabet Soup Charity

## 📘 Overview

The purpose of this analysis is to create a **binary classifier** using deep learning that can predict whether an applicant will be **successful** in securing funding from the nonprofit **Alphabet Soup**. The classifier is based on a historical dataset of funded organizations, containing various features such as application type, organizational classification, income level, and special considerations.

This tool is intended to support Alphabet Soup's decision-making process when reviewing new applications for funding.

---

## 📂 Files Included

- `AlphabetSoupCharity.ipynb` — Initial preprocessing, modeling, and evaluation
- `AlphabetSoupCharity_Optimization.ipynb` — Optimized model version with enhancements
- `AlphabetSoupCharity.h5` — Initial saved model
- `AlphabetSoupCharity_Optimization.h5` — Optimized model saved
- `charity_data.csv` — Source dataset (from cloud source)

---

## 🔍 Data Preprocessing

### ✅ Target Variable
- **`IS_SUCCESSFUL`** — Binary target:
  - `1`: Successful funding
  - `0`: Unsuccessful

### ✅ Features (Examples)
- `APPLICATION_TYPE`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`

### ❌ Columns Removed
- `EIN` — Identifier only
- `NAME` — Organization names are not predictive

### 🧹 Cleaning Steps
- Combined rare categories (less than N observations) into `"Other"` for categorical variables
- Applied `pd.get_dummies()` for one-hot encoding
- Scaled numerical values with `StandardScaler`
- Split dataset into training and test subsets using `train_test_split`

---

## 🤖 Initial Neural Network Model (AlphabetSoupCharity.ipynb)

### Model Architecture
- **Input Layer**: 43 input features
- **Hidden Layers**:
  - Layer 1: 80 neurons, ReLU
  - Layer 2: 30 neurons, ReLU
- **Output Layer**: 1 neuron, Sigmoid activation (binary classification)

### Training
- Optimizer: Adam
- Epochs: 100
- Loss function: Binary cross-entropy

### Performance Results
- **Accuracy**: ~72%
- **Loss**: ~0.55
- **HDF5 File**: `AlphabetSoupCharity.h5`

---

## 🛠️ Optimized Model (AlphabetSoupCharity_Optimization.ipynb)

### Optimization Techniques Applied
- Increased neurons in hidden layers (e.g., 120 and 60)
- Added a third hidden layer
- Reduced rare category thresholds for fewer "Other" encodings
- Increased training epochs to 150
- Tested tanh and relu activation combinations

### Optimized Model Architecture
- **Input Layer**: 43 features
- **Hidden Layers**:
  - Layer 1: 120 neurons, ReLU
  - Layer 2: 60 neurons, ReLU
  - Layer 3: 30 neurons, ReLU
- **Output Layer**: 1 neuron, Sigmoid

### Optimized Performance
- **Accuracy**: **~76.5%**
- **Loss**: ~0.47
- **HDF5 File**: `AlphabetSoupCharity_Optimization.h5`

---

## 🧾 Results Summary

### Did we hit the 75% accuracy goal?
✅ **Yes**, after optimization, accuracy improved to exceed the 75% threshold.

### Key Takeaways
- **ASK_AMT** and **INCOME_AMT** had strong influence on success probability.
- Encoding rare categorical values helped prevent overfitting.
- Adjusting network architecture was more impactful than changing activation functions alone.

### Recommendation
Although the deep learning model achieved decent performance, other models like:
- **Random Forest Classifier**
- **Gradient Boosting Classifier (e.g., XGBoost)**  
might perform better given their strength in structured/tabular data and class imbalance handling.

---

## 💡 How to Run

1. Clone the repo:
git clone https://github.com/yourusername/deep-learning-challenge.git
cd deep-learning-challenge
2.	Launch Jupyter or upload the notebooks to Google Colab
3.	Install required libraries: pip install tensorflow pandas scikit-learn matplotlib

---

## 👩‍💻 Author

Aditi Nankar
