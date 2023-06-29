import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix
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

# Text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_train['processed_text'])

# Convert labels to binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df_train['clean_annotation'].apply(lambda x: ast.literal_eval(x)))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Uncomment for GridSearch
#parameters = {'estimator__C':[1, 10, 100, 1000],
#              'estimator__gamma':[0.1, 0.01, 0.001, 0.0001],
#              'estimator__kernel':['rbf']}
#
#svc = OneVsRestClassifier(SVC())
#clf = GridSearchCV(svc, parameters, scoring='recall_micro', cv=5)
#clf.fit(X_train, y_train)
#
# best parameters
#print("Best Parameters: ", clf.best_params_)


# Fit TfidfVectorizer on training data
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df_train['processed_text'])

# Transform training and test data
#X_train = vectorizer.transform(df_train['processed_text'])
X_test = vectorizer.transform(df_test['text'])
y_test = mlb.fit_transform(df_test['clean_annotation'].apply(lambda x: ast.literal_eval(x)))


# Use the same model for training and prediction
model = OneVsRestClassifier(SVC(kernel='rbf', C=1000, gamma=0.01))
model.fit(X_train, y_train)

# Predict labels for test data
df_test['predictions'] = mlb.inverse_transform(model.predict(X_test))
df_test.to_csv('../data/predictions/SVM_aspects_test_withDA.csv', sep=';')
predictions = model.predict(X_test)

report = classification_report(y_test, predictions, target_names=mlb.classes_, output_dict=True)
print(report)

rounded_dict = {}
for key, inner_dict in report.items():
    rounded_inner_dict = {}
    for inner_key, value in inner_dict.items():
        rounded_inner_dict[inner_key] = round(value, 2)
    rounded_dict[key] = rounded_inner_dict

print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv('../classification_reports/classification_report_test_SVM_aspects_withoutDA.csv', sep=';')