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
df_test['processed_text'] = df_test['text'].apply(preprocess_text)

# Combine processed text and processed aspect
df_train['processed_combined'] = df_train['processed_text'] + ' ' + df_train['label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df_train['processed_combined'], df_train['sentiment'], test_size=0.2, random_state=42)

# Vectorize the combined text and aspect using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Build and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train_vec, y_train)

# Make predictions on the validation set
y_val_pred = mlp.predict(X_val_vec)

# Print classification report
print(classification_report(y_val, y_val_pred))

report = classification_report(y_val, y_val_pred, output_dict=True)
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
df_output.to_csv('../classification_reports/classification_report_validation_MLP_sentiments_withoutDA.csv', sep=';')

# Combine processed text and processed aspect
df_test['processed_combined'] = df_test['processed_text'] + ' ' + df_test['label']

# Vectorize the combined text and aspect using the same TF-IDF vectorizer
X_test_vec = vectorizer.transform(df_test['processed_combined'])

# Use the trained MLP model to make predictions on X_test_vec
y_test_pred = mlp.predict(X_test_vec)
df_test['predictions'] = y_test_pred
df_test.to_csv("../data/predictions/MLP_sentiments_withoutDA.csv", sep=";")
print(classification_report(df_test['sentiment'].to_list(), y_test_pred))

report = classification_report(df_test['sentiment'].to_list(), y_test_pred, output_dict=True)
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
df_output.to_csv('../classification_reports/classification_report_test_MLP_sentiments_withoutDA.csv', sep=';')