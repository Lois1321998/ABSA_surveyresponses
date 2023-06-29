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

start_time = time.time()

class BinarySentimentClassifier:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict(self, sentence, aspect, sentiment_labels):
        inputs = self.tokenizer([f"{sentence} [SEP] {aspect} [SEP] {label}" for label in sentiment_labels], 
                                 padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=1)[:, 1].tolist()

        # Choose the label with the highest probability
        predicted_sentiment = sentiment_labels[probabilities.index(max(probabilities))]

        return predicted_sentiment

def calculate_f1_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print()
    return f1_score(y_true, y_pred, average='binary', pos_label='positief')

def transform_sentiment_data(df, annotations='common_annotation'):
    # Create a new dataframe to store the transformed data
    new_data = []

    # Iterate over each row in the original dataframe
    for index, row in df.iterrows():
        text = row['text']
        labels = row[annotations]
        clean_annotation = row['clean_annotation']
        
        # Iterate over each label in the row
        for label in ast.literal_eval(labels):
            #print(label)
            new_row = {'text': text, 'label': label, 'sentiment': [label[-3:] if label != 'niks' else ''], 'common_annotation': labels, 'clean_annotation': clean_annotation}
            new_data.append(new_row)

    # Create the transformed dataframe
    transformed_df = pd.DataFrame(new_data)
    return transformed_df

df_train = pd.read_csv("../data/zeroshot_classification/preprocessed_train_ABSA_withoutDA.csv", sep=';')
df_test = pd.read_csv("../data/zeroshot_classification/preprocessed_test_ABSA.csv", sep=';')

df_train = transform_sentiment_data(df_train)
df_test = transform_sentiment_data(df_test)

df_train['label'] = df_train['label'].str.rstrip('_NEG').str.rstrip('_POS').replace('persoonlijke-aandacht', 'persoonlijke aandacht').replace('roosters-planning', 'roosters en planning')
df_train['sentiment'] = df_train['sentiment'].apply(lambda x: x[0]).replace('NEG', 'negatief').replace('POS', 'positief')
df_train = df_train[df_train['sentiment'].apply(lambda x: x != '')]

df_test['label'] = df_test['label'].str.rstrip('_NEG').str.rstrip('_POS').replace('persoonlijke-aandacht', 'persoonlijke aandacht').replace('roosters-planning', 'roosters en planning')
df_test['sentiment'] = df_test['sentiment'].apply(lambda x: x[0]).replace('NEG', 'negatief').replace('POS', 'positief')
df_test = df_test[df_test['sentiment'].apply(lambda x: x != '')]

# Initialize the classifier with the specified model
model_name = 'wietsedv/bert-base-dutch-cased'
sentiment_labels = ['positief', 'negatief']
classifier = BinarySentimentClassifier(model_name)

# Get the predicted sentiment for each row
df_train["predicted_sentiment"] = df_train.apply(lambda row: classifier.predict(row["text"], row["common_annotation"], sentiment_labels), axis=1)

# Calculate the F1-score
f1 = calculate_f1_score(df_train["sentiment"].tolist(), df_train["predicted_sentiment"].tolist())
print("F1-score:", f1)

y_true = df_train['sentiment']
y_pred = df_train['predicted_sentiment']

# Print the classification report
print(classification_report(y_true, y_pred))


# Print the classification report
report = classification_report(y_true, y_pred, output_dict=True)
print(report)

rounded_dict = {}
for key, value in report.items():
    if isinstance(value, dict):
        rounded_inner_dict = {inner_key: round(inner_value, 2) for inner_key, inner_value in value.items()}
        rounded_dict[key] = rounded_inner_dict
    else:
        rounded_dict[key] = round(value, 2)

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_train_BERTje_zeroshot_sentiments_withoutDA.csv', sep=';')

# Initialize the classifier with the specified model
model_name = 'wietsedv/bert-base-dutch-cased'
sentiment_labels = ['positief', 'negatief']
classifier = BinarySentimentClassifier(model_name)

# Get the predicted sentiment for each row
df_test["predicted_sentiment"] = df_test.apply(lambda row: classifier.predict(row["text"], row["common_annotation"], sentiment_labels), axis=1)

# Calculate the F1-score
f1 = calculate_f1_score(df_test["sentiment"].tolist(), df_test["predicted_sentiment"].tolist())
print("F1-score:", f1)

df_test.to_csv("../data/predictions/BERTje_zeroshot_sentiments_withoutDA.csv", sep=';')

y_true = df_test['sentiment']
y_pred = df_test['predicted_sentiment']

# Print the classification report
print(classification_report(y_true, y_pred, target_names=['positief', 'negatief']))


# Print the classification report
report = classification_report(y_true, y_pred, target_names=['positief', 'negatief'], output_dict=True)
print(report)

rounded_dict = {}
for key, value in report.items():
    if isinstance(value, dict):
        rounded_inner_dict = {inner_key: round(inner_value, 2) for inner_key, inner_value in value.items()}
        rounded_dict[key] = rounded_inner_dict
    else:
        rounded_dict[key] = round(value, 2)

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_test_BERTje_zeroshot_sentiments_withoutDA.csv', sep=';')

print("--- %s seconds ---" % (time.time() - start_time))