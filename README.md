# Titanic - Machine Learning from Disaster (TensorFlow Decision Forests)

This project is a solution to the classic Titanic survival prediction challenge on [Kaggle](https://www.kaggle.com/competitions/titanic). It uses **TensorFlow Decision Forests (TFDF)** with custom preprocessing, feature engineering, hyperparameter tuning, and ensemble techniques to build robust models.

## 📁 Project Structure

- `titanic_tfdf_cleaned.py`: Cleaned, modular, and professional Python implementation.
- `submission.csv`: Prediction file for Kaggle submission.
- `train.csv`, `test.csv`: Original dataset files (from Kaggle).

## 🧪 Features & Techniques Used

- **Preprocessing**:

  - Custom name cleaning
  - Ticket decomposition into number & item prefix

- **Modeling**:

  - `GradientBoostedTreesModel` from TFDF
  - Feature control with `FeatureUsage`

- **Hyperparameter Tuning**:

  - `tfdf.tuner.RandomSearch` with control over strategies (`LOCAL`, `BEST_FIRST_GLOBAL`) and parameters (depth, shrinkage, axis-alignment)

- **Ensembling**:

  - 100-model soft voting ensemble with varying random seeds

## 🧰 Dependencies

```bash
pip install pandas numpy tensorflow tensorflow_decision_forests
```

## 🚀 How to Run

```bash
# 1. Load and preprocess the data
# 2. Train base model and evaluate
# 3. Tune model with custom parameters or random search
# 4. Generate predictions for submission
```

You can run it in Google Colab or locally with a Python environment that supports TFDF.

## 📈 Evaluation

Model accuracy and loss are printed after each stage (baseline, tuned, ensemble). Submission is generated with probability thresholding (>= 0.5).

## 📤 Submission Format

The submission file (`submission.csv`) will contain two columns:

```
PassengerId,Survived
892,0
893,1
...
```

## 📜 License

This project is intended for educational and non-commercial use only.

---

Built using TensorFlow Decision Forests and inspired by open-source data science workflows.
