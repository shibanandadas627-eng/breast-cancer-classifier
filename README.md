# 🔬 Breast Cancer Diagnosis Classifier

An end-to-end machine learning project that predicts whether a breast tumour is **benign** or **malignant** from cell-nucleus measurements taken from a digitised biopsy image.

> ⚠️ **Educational project only.** This is not a medical device and must not be used for real diagnosis.

## 📌 Problem Statement
Early and accurate detection of malignant tumours is critical. Given 30 numeric features describing cell nuclei, build a model that classifies tumours as benign (0) or malignant (1) while keeping **false negatives (missed cancers) as low as possible**.

## 📊 Dataset
- **Breast Cancer Wisconsin (Diagnostic)** dataset, bundled with scikit-learn (no download needed).
- 569 samples, 30 numeric features (mean, standard error and "worst" values of radius, texture, perimeter, area, smoothness, etc.).
- Classes: 357 benign (62.7%), 212 malignant (37.3%). No missing values.

## 🧰 Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

## 🗂️ Project Structure
```
breast-cancer-classifier/
├── src/
│   ├── data.py        # load + stratified split
│   ├── eda.py         # exploratory analysis -> reports/figures
│   ├── train.py       # CV comparison, tuning, evaluation, saves model
│   └── predict.py     # inference (sample or your own CSV)
├── tests/test_pipeline.py
├── models/best_model.joblib
├── reports/
│   ├── metrics.json
│   └── figures/
├── requirements.txt
├── LICENSE
└── README.md
```

## 🚀 Getting Started
```bash
git clone https://github.com/<your-username>/breast-cancer-classifier.git
cd breast-cancer-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda       # generate EDA plots
python -m src.train     # train, tune, evaluate, save model
python -m src.predict   # demo predictions (or: --csv your_file.csv)
python -m unittest discover -s tests -t .   # run tests
```

## 🔎 Methodology
1. **EDA** – class balance, feature correlations, distributions of the most discriminative features.
2. **Preprocessing** – `StandardScaler` inside a scikit-learn `Pipeline` (prevents data leakage during CV).
3. **Split** – 80/20 stratified train/test; the test set is only used once, at the end.
4. **Model comparison** – Logistic Regression, SVM (RBF), Random Forest with 5-fold stratified CV (ROC-AUC).
5. **Tuning** – `GridSearchCV` per model; best model chosen by CV ROC-AUC.
6. **Evaluation** – accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importance.

## 📈 Results

**5-fold CV ROC-AUC (training data, after tuning)**

| Model | CV ROC-AUC |
|---|---|
| **SVM (RBF)** (C=100, gamma=0.001) | **0.9960** |
| Logistic Regression (C=1) | 0.9958 |
| Random Forest | 0.9880 |

**Final test-set performance (SVM, 114 unseen samples)**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 97.4% | 100% | 92.9% | 96.3% | 0.996 |

The model produced zero false positives and missed 3 of 42 malignant cases. Since missed cancers are the costlier error, a natural next step is lowering the decision threshold to trade some precision for higher recall.

| Confusion matrix | ROC curve |
|---|---|
| ![cm](reports/figures/confusion_matrix.png) | ![roc](reports/figures/roc_curve.png) |

| Feature importance | Correlations |
|---|---|
| ![fi](reports/figures/feature_importance.png) | ![corr](reports/figures/correlation_heatmap.png) |

## 💡 Key Insights
- Size- and shape-related features (radius, perimeter, area, concavity of the "worst" nuclei) are the strongest indicators of malignancy.
- Many features are highly correlated, so simple linear models perform on par with more complex ones.

## 🔮 Future Work
- Threshold tuning / cost-sensitive learning to maximise recall
- SHAP explanations for individual predictions
- Streamlit web app for interactive predictions
- Validate on an external dataset

## 📄 License
MIT – see [LICENSE](LICENSE).
