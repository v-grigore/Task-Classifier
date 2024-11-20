import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import re
from pathlib import Path
import lightgbm as lgb  # Importing LightGBM
import xgboost as xgb  # Importing XGBoost

INPUT_DIR = Path("/usr/src/app/InputData")
OUTPUT_DIR = Path("/usr/src/app/output")
DATA_DIR = Path(".")

labels = {}


# Load Data with Extended Text Extraction
def load_data(json_file_path):
    texts, labels, filenames = [], [], []  # Add filenames list
    with open(json_file_path, "r") as file:
        data_list = json.load(file)
        for data in data_list:
            text = ""
            if "event" in data:
                event = data["event"]
                text += " ".join(
                    [str(event.get(k, "")) for k in event if isinstance(event[k], str)]
                )
            if "file" in data:
                file_info = data["file"]
                text += " " + " ".join(
                    [
                        str(file_info.get(k, ""))
                        for k in file_info
                        if isinstance(file_info[k], str)
                    ]
                )
            if "message" in data:
                text += " " + data["message"]
            if "powershell" in data and "command" in data["powershell"]:
                ps_command = data["powershell"]["command"]
                text += (
                    " "
                    + str(ps_command.get("name", ""))
                    + " "
                    + " ".join(
                        [
                            str(inv.get("value", ""))
                            for inv in ps_command.get("invocation_details", [])
                        ]
                    )
                )

            label = data.get("label")
            filename = data.get("filename")  # Extract filename
            if label in [0, 1]:
                texts.append(clean_text(text))
                labels.append(label)
                filenames.append(filename)  # Add filename to the list

    return texts, labels, filenames


# Text Cleaning
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\b\d+\b", "", text)  # Remove standalone numbers
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    return text.strip()


# Preprocess and Train Model using Voting Classifier
def preprocess(texts, labels):
    # Define LightGBM and XGBoost models
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
        subsample=0.8,
        use_label_encoder=False,
    )

    # Create Voting Classifier with soft voting (average of predicted probabilities)
    voting_clf = VotingClassifier(
        estimators=[("lgb", lgb_model), ("xgb", xgb_model)], voting="soft"
    )

    # Pipeline with Tfidf Vectorizer and Voting Classifier
    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(max_features=6000, ngram_range=(1, 2))),
            ("voting_classifier", voting_clf),
        ]
    )

    # Train the model
    pipeline.fit(texts, labels)
    return pipeline


# Evaluate the Model
def evaluate_model(model, X_test, y_test, texts_test, filenames_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Store the predicted labels along with filenames
    for i in range(len(y_test)):
        labels[filenames_test[i]] = int(y_pred[i])

    # Save the classified labels
    with open(Path(OUTPUT_DIR) / "labels", "w") as outfile:
        json.dump(labels, outfile)


# Main Execution
train_file = "train.json"
test_file = "test.json"

# Load the data
train_texts, train_labels, train_filenames = load_data(train_file)
test_texts, test_labels, test_filenames = load_data(test_file)

# Print the number of training and test files
print(f"Number of training files: {len(train_filenames)}")
print(f"Number of test files: {len(test_filenames)}")

# Preprocess and train the model
model = preprocess(train_texts, train_labels)

# Evaluate the model with filenames
evaluate_model(model, test_texts, test_labels, test_texts, test_filenames)
