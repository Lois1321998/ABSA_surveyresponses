import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification

import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import time

df_train = pd.read_csv("../data/zeroshot_classification/preprocessed_train_ABSA_withoutDA.csv", sep=';', index_col=0)
df_test = pd.read_csv("../data/zeroshot_classification/preprocessed_test_ABSA.csv", sep=';', index_col=0)

df_train['common_annotation'] = df_train['common_annotation'].apply(lambda x: ast.literal_eval(x))
df_train['clean_annotation'] = df_train['clean_annotation'].apply(lambda x: ast.literal_eval(x))
df_test['common_annotation'] = df_test['common_annotation'].apply(lambda x: ast.literal_eval(x))
df_test['clean_annotation'] = df_test['clean_annotation'].apply(lambda x: ast.literal_eval(x))

class ZeroShotMultilabelClassifier:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, sentence, aspect_labels, threshold=0.5):
        inputs = self.tokenizer([f"{sentence} [SEP] {label}" for label in aspect_labels], 
                                 padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=1)[:, 1].tolist()

        # Filter aspectcs based on threshold
        predicted_aspects = [aspect_labels[i] for i, prob in enumerate(probabilities) if prob >= threshold]

        return probabilities, predicted_aspects

def predict_aspects_with_threshold(row, classifier, aspect_labels, threshold):
    probabilities, predicted_aspects = classifier.predict(row["text"], aspect_labels, threshold)
    #print(probabilities)
    return predicted_aspects

def calculate_f1_score(y_true, y_pred, mlb):
    y_true_binarized = mlb.transform(np.array(y_true))
    y_pred_binarized = mlb.transform(np.array(y_pred))
    print(classification_report(y_true_binarized, y_pred_binarized, target_names=mlb.classes_))
    print()
    return f1_score(y_true_binarized, y_pred_binarized, average='macro')

def grid_search_threshold(df, model_name, aspect_labels, thresholds):
    # Initialize the classifier with the specified model
    classifier = ZeroShotMultilabelClassifier(model_name)
    
    # Fit the MultiLabelBinarizer on the aspect labels
    mlb = MultiLabelBinarizer()
    mlb.fit([aspect_labels])



    best_f1 = 0
    best_threshold = None

    for threshold in thresholds:
        # Get the predicted aspects for each row
        df["predicted_aspects"] = df.apply(predict_aspects_with_threshold, axis=1, classifier=classifier, aspect_labels=aspect_labels, threshold=threshold)
        # Calculate the F1-score

        f1 = calculate_f1_score(df["clean_annotation"].tolist(), df["predicted_aspects"].tolist(), mlb)
        print(threshold, f1)
        # Update the best threshold and F1-score if necessary
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            break

    return best_threshold, best_f1

# Define the BERTje model name and aspect labels
model_name = 'wietsedv/bert-base-dutch-cased'
aspect_labels = ['contact','persoonlijke aandacht', 'salaris', 'communicatie','roosters en planning', 'afspraken']

# Define the thresholds to search over
thresholds = np.linspace(0.45, 0.65, 9)

best_threshold, best_f1 = grid_search_threshold(df_train, model_name, aspect_labels, thresholds)

print(f"Best threshold: {best_threshold}")
print(f"Best F1-score: {best_f1}")

# Initialize the classifier with the best threshold and specified model name
classifier = ZeroShotMultilabelClassifier(model_name)

# Get the predicted aspects for each row using the best threshold
df_train["predicted_aspects"] = df_train.apply(predict_aspects_with_threshold, axis=1, classifier=classifier, aspect_labels=aspect_labels, threshold=best_threshold)
df_test["predicted_aspects"] = df_test.apply(predict_aspects_with_threshold, axis=1, classifier=classifier, aspect_labels=aspect_labels, threshold=best_threshold)

# Print the resulting dataframe with predicted aspects
print(df_test[["text", "predicted_aspects"]])

df_test.to_csv("../data/predictions/BERTje_zeroshot_aspects_withoutDA.csv", sep=';')

mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(df_train['clean_annotation'])
y_pred = mlb.transform(df_train['predicted_aspects'])

# Print the classification report
print(classification_report(y_true, y_pred, target_names=mlb.classes_))


# Print the classification report
report = classification_report(y_true, y_pred, target_names=mlb.classes_, output_dict=True)
print(report)

rounded_dict = {}
for key, inner_dict in report.items():
    rounded_inner_dict = {}
    for inner_key, value in inner_dict.items():
        rounded_inner_dict[inner_key] = round(value, 2)
    rounded_dict[key] = rounded_inner_dict

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_train_BERTje_zeroshot_aspects_withoutDA.csv', sep=';')

mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(df_test['clean_annotation'])
y_pred = mlb.transform(df_test['predicted_aspects'])

# Print the classification report
print(classification_report(y_true, y_pred, target_names=mlb.classes_))

# Print the classification report
report = classification_report(y_true, y_pred, target_names=mlb.classes_, output_dict=True)
#print(report)

rounded_dict = {}
for key, inner_dict in report.items():
    rounded_inner_dict = {}
    for inner_key, value in inner_dict.items():
        rounded_inner_dict[inner_key] = round(value, 2)
    rounded_dict[key] = rounded_inner_dict

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_test_BERTje_zeroshot_aspects_withoutDA.csv', sep=';')

print("--- %s seconds ---" % (time.time() - start_time))