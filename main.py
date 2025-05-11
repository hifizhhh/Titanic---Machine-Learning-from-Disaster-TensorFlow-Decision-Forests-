# Titanic - Machine Learning from Disaster (TFDF Version)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

# ============================
# 1. Data Loading
# ============================
df_train = pd.read_csv("/content/train.csv")
df_test = pd.read_csv("/content/test.csv")


# ============================
# 2. Preprocessing
# ============================
def preprocess(df):
    df = df.copy()

    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

    def ticket_number(x):
        return x.split(" ")[-1]

    def ticket_item(x):
        items = x.split(" ")
        return "NONE" if len(items) == 1 else "_".join(items[:-1])

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    return df


preprocessed_df_train = preprocess(df_train)
preprocessed_df_test = preprocess(df_test)

# ============================
# 3. Feature Selection
# ============================
input_features = list(preprocessed_df_train.columns)
for col in ["PassengerId", "Ticket", "Survived"]:
    input_features.remove(col)


# ============================
# 4. TensorFlow Dataset Conversion
# ============================
def tokenize_names(features, labels=None):
    features["Name"] = tf.strings.split(features["Name"])
    return features, labels


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    preprocessed_df_train, label="Survived"
).map(tokenize_names)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_df_test).map(
    tokenize_names
)

# ============================
# 5. Initial Model Training
# ============================
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,
    random_seed=1234,
)
model.fit(train_ds)

self_evaluation = model.make_inspector().evaluation()
print(
    f"Initial Model - Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}"
)

# ============================
# 6. Tuned Model with Custom Hyperparameters
# ============================
model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True,
    min_examples=1,
    categorical_algorithm="RANDOM",
    max_depth=4,
    shrinkage=0.05,
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="MIN_MAX",
    sparse_oblique_num_projections_exponent=2.0,
    num_trees=2000,
    random_seed=1234,
)
model.fit(train_ds)

self_evaluation = model.make_inspector().evaluation()
print(
    f"Tuned Model - Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}"
)


# ============================
# 7. Prediction & Submission
# ============================
def prediction_to_kaggle_format(model, threshold=0.5):
    proba_survive = model.predict(test_ds, verbose=0)[:, 0]
    return pd.DataFrame(
        {
            "PassengerId": df_test["PassengerId"],
            "Survived": (proba_survive >= threshold).astype(int),
        }
    )


def make_submission(kaggle_predictions):
    path = "/content/Kaggle/submission.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kaggle_predictions.to_csv(path, index=False)
    print(f"Submission exported to {path}")


kaggle_predictions = prediction_to_kaggle_format(model)
make_submission(kaggle_predictions)

# ============================
# 8. Hyperparameter Tuning (Random Search)
# ============================
tuner = tfdf.tuner.RandomSearch(num_trials=1000)
tuner.choice("min_examples", [2, 5, 7, 10])
tuner.choice("categorical_algorithm", ["CART", "RANDOM"])

local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [3, 4, 5, 6, 8])

global_search_space = tuner.choice(
    "growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True
)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

tuner.choice("shrinkage", [0.02, 0.05, 0.10, 0.15])
tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])

tuner.choice("split_axis", ["AXIS_ALIGNED"])
oblique_space = tuner.choice("split_axis", ["SPARSE_OBLIQUE"], merge=True)
oblique_space.choice(
    "sparse_oblique_normalization", ["NONE", "STANDARD_DEVIATION", "MIN_MAX"]
)
oblique_space.choice("sparse_oblique_weight", ["BINARY", "CONTINUOUS"])
oblique_space.choice("sparse_oblique_num_projections_exponent", [1.0, 1.5])

# Train with tuner
model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
model.fit(train_ds, verbose=0)

tuned_self_evaluation = model.make_inspector().evaluation()
print(
    f"Tuned Model (Random Search) - Accuracy: {tuned_self_evaluation.accuracy} Loss: {tuned_self_evaluation.loss}"
)

# ============================
# 9. Ensemble of 100 Models (Soft Voting)
# ============================
predictions = None
num_predictions = 0

for i in range(100):
    print(f"Training ensemble model {i}")
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=0,
        features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
        exclude_non_specified_features=True,
        random_seed=i,
        honest=True,
    )
    model.fit(train_ds)

    sub_predictions = model.predict(test_ds, verbose=0)[:, 0]
    predictions = (
        sub_predictions if predictions is None else predictions + sub_predictions
    )
    num_predictions += 1

predictions /= num_predictions

kaggle_predictions = pd.DataFrame(
    {
        "PassengerId": df_test["PassengerId"],
        "Survived": (predictions >= 0.5).astype(int),
    }
)
make_submission(kaggle_predictions)
