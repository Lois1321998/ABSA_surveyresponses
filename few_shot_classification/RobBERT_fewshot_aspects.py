import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import ast
import matplotlib.pyplot as plt

df_train = pd.read_csv("../data/zeroshot_classification/preprocessed_train_ABSA_withoutDA.csv", sep=';', index_col=0)
df_test = pd.read_csv("../data/zeroshot_classification/preprocessed_test_ABSA.csv", sep=';', index_col=0)
df_augtrain = pd.read_csv("../data/zeroshot_classification/preprocessed_train_ABSA_withDA.csv", sep=';', index_col=0)

df_train['common_annotation'] = df_train['common_annotation'].apply(lambda x: ast.literal_eval(x))
df_train['clean_annotation'] = df_train['clean_annotation'].apply(lambda x: ast.literal_eval(x))
df_test['common_annotation'] = df_test['common_annotation'].apply(lambda x: ast.literal_eval(x))
df_test['clean_annotation'] = df_test['clean_annotation'].apply(lambda x: ast.literal_eval(x))
df_augtrain['common_annotation'] = df_augtrain['common_annotation'].apply(lambda x: ast.literal_eval(x))
df_augtrain['clean_annotation'] = df_augtrain['clean_annotation'].apply(lambda x: ast.literal_eval(x))
#df_augtrain['text'] = df_augtrain['text'].apply(lambda x: ast.literal_eval(x))

classes = ['contact','persoonlijke aandacht', 'salaris', 'communicatie','roosters en planning', 'afspraken']

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=classes)

# Transform the 'clean_annotation' column and add to the original DataFrame
df_transformed = pd.DataFrame(mlb.fit_transform(df_train['clean_annotation']),
                              columns=classes,
                              index=df_train.index)
df_train['onehot'] = df_transformed[df_transformed.columns].values.tolist()

# Transform the 'clean_annotation' column and add to the original DataFrame
df_transformed_2 = pd.DataFrame(mlb.fit_transform(df_test['clean_annotation']),
                              columns=classes,
                              index=df_test.index)

df_test['onehot'] = df_transformed_2[df_transformed_2.columns].values.tolist()

# Transform the 'clean_annotation' column and add to the original DataFrame
df_transformed_3 = pd.DataFrame(mlb.fit_transform(df_augtrain['clean_annotation']),
                              columns=classes,
                              index=df_augtrain.index)

df_augtrain['onehot'] = df_transformed_3[df_transformed_3.columns].values.tolist()


# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8 #8
VALID_BATCH_SIZE = 4 #4
TEST_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base')

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.onehot
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
train_size = 0.8
train_dataset=df_train.sample(frac=train_size,random_state=200)
validation_dataset=df_train.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = df_test.reset_index(drop=True)
augtrain_dataset = df_augtrain.reset_index(drop=True)


print("FULL Dataset: {}".format(df_train.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VALIDATION Dataset: {}".format(validation_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))
print("AUGMENTED TRAIN Dataset: {}".format(augtrain_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(validation_dataset, tokenizer, MAX_LEN)
test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
augtrain_set = CustomDataset(augtrain_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **validation_params)
test_loader = DataLoader(test_set, **test_params)
augtrain_loader =  DataLoader(augtrain_set, **train_params)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        #self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l1 = transformers.RobertaModel.from_pretrained('pdelobelle/robbert-v2-dutch-base', return_dict = False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch, loader):
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    for _,data in enumerate(loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        # compute accuracy
        outputs = torch.sigmoid(outputs)  # apply sigmoid to get values between 0 and 1
        preds = (outputs > 0.5).int()  # anything above 0.5 is considered a positive prediction

        accuracy = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())

        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_losses, epoch_accuracies



losses = []
accuracies = []
for epoch in range(EPOCHS):
    epoch_losses, epoch_accuracies = train(epoch, training_loader)
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
    losses.append(avg_loss)
    accuracies.append(avg_accuracy)
    print(f'Epoch: {epoch}, Loss: {avg_loss}, Accuracy: {avg_accuracy}')
    
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
    

def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(classification_report(targets, outputs, target_names=mlb.classes_))
    print()
    

report = classification_report(targets, outputs, target_names=mlb.classes_, output_dict=True)
rounded_dict = {}
for key, inner_dict in report.items():
    rounded_inner_dict = {}
    for inner_key, value in inner_dict.items():
        rounded_inner_dict[inner_key] = round(value, 2)
    rounded_dict[key] = rounded_inner_dict

rounded_dict = dict(sorted(rounded_dict.items()))
print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
#df_output.to_csv("../classification_reports/RobBERT_fewshot_aspects_validation_withoutDA.csv", sep=';')

def predict(texts, tokenizer, model, max_len=512):
    model.eval()

    predictions = []

    for text in texts:
        # Prepare the text data similarly as you did for your training data
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
        mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
        token_type_ids = torch.tensor([inputs['token_type_ids']], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids)

        # Apply sigmoid function to convert logits into probabilities (between 0 and 1)
        outputs = torch.sigmoid(outputs).cpu().numpy()

        # Convert probabilities into binary predictions (above 0.5 is considered a positive prediction)
        preds = (outputs > 0.5).astype(int)

        predictions.append(preds)

    return predictions

texts = df_test['text'].tolist()
predictions = predict(texts, tokenizer, model, MAX_LEN)

df_test['predictions'] = predictions

predictions_2d = np.vstack(predictions)
decoded_predictions = mlb.inverse_transform(predictions_2d)

#df_test['predictions'] = df_test['predictions'].apply(lambda x: x[0])
df_test['decoded_predictions'] = decoded_predictions
df_test.to_csv("../data/predictions/RobBERT_fewshot_aspects_withoutDA.csv", sep=';')

print(classification_report(df_test['onehot'].tolist(), df_test['predictions'].tolist(), target_names=mlb.classes_))
report = classification_report(df_test['onehot'].tolist(), df_test['predictions'].tolist(), target_names=mlb.classes_, output_dict=True)
rounded_dict = {}
for key, inner_dict in report.items():
    rounded_inner_dict = {}
    for inner_key, value in inner_dict.items():
        rounded_inner_dict[inner_key] = round(value, 2)
    rounded_dict[key] = rounded_inner_dict
rounded_dict = dict(sorted(rounded_dict.items()))
print(rounded_dict)

df_output = pd.DataFrame(rounded_dict).transpose()
df_output.to_csv("../classification_reports/RobBERT_fewshot_aspects_test_withoutDA.csv", sep=';')


print(classification_report(df_test['onehot'].tolist(), df_test['predictions'].tolist(), target_names=mlb.classes_))