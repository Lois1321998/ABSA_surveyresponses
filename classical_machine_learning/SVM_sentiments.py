import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import ast

from sklearn.model_selection import GridSearchCV

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

df_train = transform_sentiment_data(df_train)
df_test = transform_sentiment_data(df_test)

df_train['label'] = df_train['label'].str.rstrip('_NEG').str.rstrip('_POS').replace('persoonlijke-aandacht', 'persoonlijke aandacht').replace('roosters-planning', 'roosters en planning')
df_train['sentiment'] = df_train['sentiment'].apply(lambda x: x[0]).replace('NEG', 'negatief').replace('POS', 'positief')
df_train = df_train[df_train['sentiment'].apply(lambda x: x != '')]

df_test['label'] = df_test['label'].str.rstrip('_NEG').str.rstrip('_POS').replace('persoonlijke-aandacht', 'persoonlijke aandacht').replace('roosters-planning', 'roosters en planning')
df_test['sentiment'] = df_test['sentiment'].apply(lambda x: x[0]).replace('NEG', 'negatief').replace('POS', 'positief')
df_test = df_test[df_test['sentiment'].apply(lambda x: x != '')]

df_train['processed_text'] = df_train['text'].apply(preprocess_text)

# Combine 'text' and 'label' fields, and apply text preprocessing
df_train['text_label'] = df_train['text'] + ' ' + df_train['label']
df_train['processed_text'] = df_train['text_label'].apply(preprocess_text)

# Convert the combined and processed text data into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_train['processed_text'])

# Convert sentiment labels into numerical format
# assuming sentiment labels are strings like 'positive', 'negative'
le = LabelEncoder()
y = le.fit_transform(df_train['sentiment'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#uncomment for GridSeach
# Define the parameter values that should be searched
#param_grid = {'C': [0.01, 0.1, 1, 10, 100], 
 #             'max_iter': [1000, 2000, 3000]}

# Instantiate the grid
#grid = GridSearchCV(LinearSVC(), param_grid, cv=5, scoring='accuracy')

# Fit the grid with data
#grid.fit(X_train, y_train)

# View the complete results
#print("Grid scores on training set:")
#means = grid.cv_results_['mean_test_score']
#stds = grid.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, grid.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Print the details of the best model and its accuracy
#print("\nBest parameters set found on development set: ", grid.best_params_)

svm = LinearSVC(C=10)
svm.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = svm.predict(X_val)
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

report = classification_report(y_val, y_val_pred, target_names=le.classes_, output_dict=True)
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
df_output.to_csv('../classification_reports/classification_report_validation_SVM_sentiments_withoutDA.csv', sep=';')

# Combine 'text' and 'label' fields, and apply text preprocessing
df_test['text_label'] = df_test['text'] + ' ' + df_test['label']
df_test['processed_text'] = df_test['text_label'].apply(preprocess_text)

# Transform the text data into vectors
X_test = vectorizer.transform(df_test['processed_text'])
y_test = le.fit_transform(df_test['sentiment'])

# Predict the sentiment
df_test['predicted_sentiment'] = le.inverse_transform(svm.predict(X_test))
df_test.to_csv('../data/predictions/SVM_sentiments_test.csv', sep=';')
predictions = svm.predict(X_test)

report = classification_report(y_test, predictions, target_names=le.classes_, output_dict=True)
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
df_output.to_csv('../classification_reports/classification_report_test_SVM_sentiments_withoutDA.csv', sep=';')