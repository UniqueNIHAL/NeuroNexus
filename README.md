# Credit Card Fraud Detection

This project uses machine learning to detect fraudulent transactions in a credit card dataset. The dataset is highly imbalanced, with very few fraudulent cases compared to genuine ones, which makes it a perfect case to demonstrate data preprocessing, normalization, class imbalance handling, and performance evaluation.

---

## 👨‍💻 Developed by:
**Nihal H U**  
BCA – Cloud Technology & Information Security  
Jain (Deemed-to-be) University

---

## 🔍 What It Does

- Loads the credit card transactions dataset
- Normalizes `Time` and `Amount` features using `StandardScaler`
- Handles class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**
- Trains a **Random Forest Classifier** to detect fraudulent vs genuine transactions
- Evaluates using **accuracy, precision, recall, F1-score**
- Saves a **confusion matrix plot** and **text-based classification report**

---

## 🛠️ How to Run

1. Make sure you have Python installed (`>=3.8` recommended)
2. Install required libraries:
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
3. Place `creditcard.csv` inside the `data/` folder.
4. Run the script:
    ```
    python fraud_detection.py
    ```

---

## 📊 Output Files

- **confusion_matrix.png** – Visual representation of model predictions
- **classification_report.txt** – Detailed performance metrics

---

## 📌 Notes

- Dataset: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Techniques Used: Data normalization, oversampling (SMOTE), ensemble learning (Random Forest)

---

