import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing lackages from our NLP-Hugging Package
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast, RobertaForSequenceClassification, BertConfig,BertModel, BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Importing wand for logging and hyper-parameter tuning
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import time

start_time = time.time()

df_train = pd.read_csv("../data/zeroshot_classification/preprocessed_train_ABSA_withoutDA.csv", sep=';')
df_test = pd.read_csv("../data/zeroshot_classification/preprocessed_test_ABSA.csv", sep=';')

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

class Preprocess:
    def __init__(self, df):
        """
        Constructor for the class
        :param df: Input Dataframe to be pre-processed
        """
        self.df = df
        self.encoded_dict = dict()

    def encoding(self, x):
        if x not in self.encoded_dict.keys():
            self.encoded_dict[x] = len(self.encoded_dict)
        return self.encoded_dict[x]

    def processing(self):
        self.df['encoded_polarity'] = self.df['sentiment'].apply(lambda x: self.encoding(x))
        #self.df.drop(['sentiment'], axis=1, inplace=True)
        return self.encoded_dict, self.df

# Creating a CustomDataset class that is used to read the updated dataframe and tokenize the text. 
# The class is used in the return_dataloader function

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.text[index])
        text = " ".join(text.split())
        aspect = str(self.data.label[index])
        aspect = " ".join(aspect.split())

        inputs = self.tokenizer.encode_plus(
            text,
            aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.encoded_polarity[index], dtype=torch.float)
        }

    def __len__(self):
        return self.len

    
# Creating a function that returns the dataloader based on the dataframe and the specified train and validation batch size. 

def return_dataloader(df, tokenizer, train_batch_size, validation_batch_size, MAX_LEN, train_size=0.7, df_test=df_test):
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = df_test.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': train_batch_size,
                'shuffle': True,
                'num_workers': 1
                }

    val_params = {'batch_size': validation_batch_size,
                    'shuffle': True,
                    'num_workers': 1
                    }
    
    test_params = {'batch_size': validation_batch_size,
                    'shuffle': True,
                    'num_workers': 1
                    }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)
    
    return training_loader, validation_loader, test_loader


# Creating the customized model, by adding a drop out and a dense layer on top of bert to get the final output for the model. 

class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.model_layer = RobertaModel.from_pretrained('pdelobelle/robbert-v2-dutch-base')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.model_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Function to return model based on the defination of Model Class
def return_model(device):
    model = ModelClass()
    model = model.to(device)
    return model



# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch, model, device, training_loader, optimizer, loss_function):
    n_correct = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask).squeeze()
        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        if _%100==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct)/nb_tr_examples 

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct)/nb_tr_examples
    print(f'The Total Accuracy for Epoch {epoch}: {epoch_accu}')
    print(f'The Total Loss for Epoch {epoch}: {epoch_loss}')

    return epoch_loss, epoch_accu


# Function to run the validation dataloader to validate the performance of the fine tuned model. 

def valid(epoch, model, device, validation_loader, loss_function):
    n_correct = 0; total = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for _,data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Append the labels and predictions for classification report
            all_targets.extend(targets.detach().cpu().numpy())
            all_predictions.extend(big_idx.detach().cpu().numpy())

            if _%100==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct)/nb_tr_examples 

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct)/nb_tr_examples

    print(f'The Validation Accuracy: {(n_correct)/nb_tr_examples}')
    
    # Now generate the classification report
    print(classification_report(all_targets, all_predictions, target_names=['negative', 'positive']))
    return all_targets, all_predictions


MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')

# Reading the dataset and pre-processing it for usage
df = df_train
pre = Preprocess(df)
encoding_dict, df = pre.processing()

# Creating the training and validation dataloader using the functions defined above
training_loader, validation_loader, test_loader = return_dataloader(df, tokenizer, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, MAX_LEN, df_test)

# Defining the model based on the function and ModelClass defined above
model = return_model(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

losses = []
accuracies = []

# Fine tuning the model using the train function:
for epoch in range(EPOCHS):
    epoch_loss, epoch_accu = train(epoch, model, device, training_loader, optimizer, loss_function)
    losses.append(epoch_loss)
    accuracies.append(epoch_accu)
    

def plot_metrics(losses, accuracies):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.plot(epochs, losses, 'bo-', label='Training loss')
    plt.plot(epochs, accuracies, 'go-', label='Training accuracy')
    #plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    plt.show()
    

targets, pred = valid(epoch, model, device, validation_loader, loss_function)

print("--- %s seconds ---" % (time.time() - start_time))

report = classification_report(targets, pred, output_dict=True)
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
df_output.to_csv('classification_report_validation_RobBERT_fewshot_sentiments_withoutDA.csv', sep=';')

def predict_sentiments(df_test, model, device, max_len=512):
    """
    Use fine-tuned model to make sentiment predictions on a new dataframe
    """
    n_correct = 0; total = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0

    # preprocess df_test in the same way as the training data
    pre = Preprocess(df_test)
    encoding_dict, df_test = pre.processing()
    #df_test = df_test.reset_index(drop=True)
    print(df_test.columns)
    test_params = {'batch_size': 2,
                    'shuffle': True,
                    'num_workers': 1
                    }
    training_loader, validation_loader, test_loader = return_dataloader(df, tokenizer, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, MAX_LEN, df_test)
    # Convert df_test to a Dataset and DataLoader
    #test_set = CustomDataset(df_test, tokenizer, max_len)
#    test_params = {'batch_size': 2, 'shuffle': False, 'num_workers': 1}
    #test_loader = DataLoader(test_set, **test_params)

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for _,data in enumerate(test_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # Append the labels and predictions for classification report
            all_targets.extend(targets.detach().cpu().numpy())
            all_predictions.extend(big_idx.detach().cpu().numpy())

            if _%100==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct)/nb_tr_examples 

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct)/nb_tr_examples

    print(f'The Validation Accuracy: {(n_correct)/nb_tr_examples}')
    
    # Now generate the classification report
    print(classification_report(all_targets, all_predictions, target_names=['negative', 'positive']))
    return all_targets, all_predictions

targets, predictions = predict_sentiments(df_test, model, device, max_len=512)

report = classification_report(targets, predictions, output_dict=True)
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
df_output.to_csv('classification_report_test_RobBERT_fewshot_sentiments_withoutDA.csv', sep=';')

df_test['predictions'] = predictions
df_test['targets'] = targets
df_test.to_csv("../data/predictions/RobBERT_fewshot_sentiments_withoutDA.csv", sep=';')