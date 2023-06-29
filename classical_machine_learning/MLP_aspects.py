import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import f1_score, classification_report
import ast
# nltk downloads
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('dutch'))
lemmatizer = WordNetLemmatizer()

df_train = pd.read_csv("../data/zeroshot_classification/preprocessed_train_ABSA_withoutDA.csv", sep=';', index_col=0)
df_test = pd.read_csv("../data/zeroshot_classification/preprocessed_test_ABSA.csv", sep=';', index_col=0)

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    # Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Remove stopwords and lemmatize
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

df_train['processed_text'] = df_train['text'].apply(preprocess_text)
df_test['processed_text'] = df_test['text'].apply(preprocess_text)

# Convert labels to binary indicators using MultiLabelBinarizer
labels = ['contact', 'persoonlijke aandacht', 'salaris', 'communicatie', 'roosters en planning', 'afspraken']
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df_train['clean_annotation'].apply(eval))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df_train['processed_text'], y, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

#uncomment for gridsearch
# Define the parameter grid for the grid search
#param_grid = {
#    'hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128)],
#    'activation': ['relu', 'tanh'],
#    'solver': ['adam', 'sgd'],
#    'max_iter': [1000]
#}

# Create the MLP classifier
#mlp = MLPClassifier(random_state=42)
#best_model = mlp
#best_model.fit(X_train_vec, y_train)

best_model = MLPClassifier(activation='relu', hidden_layer_sizes=(256,128), solver='adam')
# Fit the model on your training data
best_model.fit(X_train_vec, y_train)

y_val_pred = best_model.predict(X_val_vec)
y_val_true = mlb.inverse_transform(y_val)
print(classification_report(y_val, y_val_pred, target_names=mlb.classes_))

report = classification_report(y_val, y_val_pred, target_names=mlb.classes_, output_dict=True)
#print(report)

rounded_dict = {}
for key, value in report.items():
    if isinstance(value, dict):
        rounded_inner_dict = {inner_key: round(inner_value, 2) for inner_key, inner_value in value.items()}
        rounded_dict[key] = rounded_inner_dict
    else:
        rounded_dict[key] = round(value, 2)

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_validation_MLP_aspects_withoutDA.csv', sep=';')

# Vectorize the preprocessed text using the same TF-IDF vectorizer
X_test_vec = vectorizer.transform(df_test['processed_text'])

y_test = mlb.fit_transform(df_test['clean_annotation'].apply(eval))

# Use the best model to make predictions on X_test_vec
y_test_pred = best_model.predict(X_test_vec)

# Convert the predicted binary indicators back to the original labels
predictions = mlb.inverse_transform(y_test_pred)
df_test['predictions'] = predictions
df_test.to_csv("../data/predictions/MLP_aspects_withoutDA.csv", sep=";")

print(classification_report(y_test, y_test_pred, target_names=mlb.classes_))

report = classification_report(y_test, y_test_pred, target_names=mlb.classes_, output_dict=True)
#print(report)

rounded_dict = {}
for key, value in report.items():
    if isinstance(value, dict):
        rounded_inner_dict = {inner_key: round(inner_value, 3) for inner_key, inner_value in value.items()}
        rounded_dict[key] = rounded_inner_dict
    else:
        rounded_dict[key] = round(value, 2)

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_test_MLP_aspects_withoutDA.csv', sep=';')