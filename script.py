# Import necessary libraries
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd



# Define the S3 path for the input CSV file
s3_path = 'your-s3-uri-to-the-csv'

# Read the dataset from the S3 path into a pandas DataFrame
df = pd.read_csv(s3_path, sep='\t',names=['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP'])

# Select only the 'TITLE' and 'CATEGORY' columns for the task
df = df[['TITLE','CATEGORY']]



# Create a dictionary to map category abbreviations to full names
my_dict = {
    'e':'Entertainment',
    'b':'Business',
    't':'Science',
    'm':'Health'
}

# Define a function to update the category names using the dictionary
def update_cat(x):
    return my_dict[x]

# Apply the function to the 'CATEGORY' column to update the names
df['CATEGORY'] = df['CATEGORY'].apply(lambda x:update_cat(x))



# This is just a tip
#df = df.sample(frac=0.05,random_state=1)

#df = df.reset_index(drop=True)
#This is where the tip ends

# Create an empty dictionary to store category encodings
encode_dict = {}



# Define a function to encode categories into numerical values
def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

# Apply the encoding function to the 'CATEGORY' column and create a new 'ENCODE_CAT' column
df['ENCODE_CAT']= df['CATEGORY'].apply(lambda x:encode_cat(x))

# resets the index of a Dataframe, which mens it creatse a new range index starting from 0 to len(df)-1, old indexes are dropped
df = df.reset_index(drop=True)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Define a custom Dataset class for handling the news data
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the NewsDataset.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            tokenizer: The tokenizer to use for text processing.
            max_len (int): The maximum length of the tokenized sequences.
        """
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input IDs, attention mask, and target tensor.
        """
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long) 
        }

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return self.len



# Define the training set size (80% of the dataset)
train_size = 0.8
# Create the training dataset by sampling from the main DataFrame
train_dataset = df.sample(frac=train_size,random_state=200)
# Create the testing dataset by dropping the training data and resetting the index
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

# Reset the index of the training dataset
train_dataset.reset_index(drop=True)


print("Full dataset: {}".format(df.shape))
print("Train dataset: {}".format(train_dataset.shape))
print("Test dataset: {}".format(test_dataset.shape))


# Define model and training parameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2



# Create NewsDataset instances for training and testing sets
training_set = NewsDataset(train_dataset,tokenizer,MAX_LEN)
testing_set = NewsDataset(test_dataset,tokenizer,MAX_LEN)


# Define parameters for the training DataLoader
train_parameters = {
                    'batch_size':TRAIN_BATCH_SIZE,
                    'shuffle':True,
                    'num_workers':0
                    }
# Define parameters for the testing DataLoader
test_parameters = {
                    'batch_size':VALID_BATCH_SIZE,
                    'shuffle':True,
                    'num_workers':0
                    }


# Create DataLoader instances for training and testing
training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)


# Define the DistilBERT model class for classification
class DistilBERTClass(torch.nn.Module):

    def __init__(self):
        """
        Initializes the DistilBERTClass model.
        """
        super(DistilBERTClass,self).__init__()

        # Load the pre-trained DistilBert model
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Define a pre-classifier linear layer
        self.pre_classifier = torch.nn.Linear(768,768)

        # Define a dropout layer for regularization
        self.dropout = torch.nn.Dropout(0.3)

        # Define the final classifier linear layer (output size 4 for 4 categories)
        self.classifier = torch.nn.Linear(768,4)

    def forward(self,input_ids, attention_mask):
        """
        Defines the forward pass of the model.

        Args:
            input_ids: The input IDs of the tokenized text.
            attention_mask: The attention mask for the input.

        Returns:
            torch.Tensor: The output logits from the classifier.
        """

        output_1 = self.l1(input_ids=input_ids,attention_mask=attention_mask)

        hidden_state = output_1[0]

        pooler = hidden_state[:,0]

        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)

        return output




# Define a function to calculate accuracy
def calculate_accu(big_idx,targets):
    n_correct = (big_idx==targets).sum().item()
    '''
    [0.88,0.1,0.33,0.7] # I love The Office 1 0 0 0 target 1 0 0 0
    [0.99,0.04,0.5,0.77] # Friends is a great show 1 0 0 0 target 1 0 0 0
    [0.38,0.12,0.1,0.88] # Elon Musk lands on Mars 0 0 0 1 target 0 0 0 1
    [0.2,00.1,.7,0.55] # Breakthrough in cancer vaccine 0 0 1 0 target 0 0 1 0
    #print(big_idx == targets) # tensor ([True, True, True, True])
    #print(big_idx == targets).sum() # tensor(4)
    print(big_idx == targets).sum().item() # 4
    '''

    return n_correct



# Define the training function
def train(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _,data in enumerate(training_loader,0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)


        loss = loss_function(outputs,targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim = 1)
        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps +=1
        nb_tr_examples +=targets.size(0)

        if _ %5000 == 0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    #print(f"The total accuracy for epoch {epoch}: {(n_correrct*100)/nb_tr_examples}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training accuracy Epoch: {epoch_accu}")

    return




# Define the validation function
def valid(epoch, model, testing_loader, device, loss_function):

    model.eval()

    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0


    with torch.no_grad():

        for _, data in enumerate(testing_loader,0):

            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask).squeeze()

            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim = 1)
            n_correct += calculate_accu(big_idx,targets)

            nb_tr_steps +=1
            nb_tr_examples += targets.size(0)

            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation loss per 1000 steps: {loss_step}")
                print(f"Validation accuracy per 1000 steps: {accu_step}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
    print(f"Validation accuracy epoch: {epoch_accu} at epoch {epoch}")

    return




# Define the main function to run the training process
def main():
    print("start")

    # Set up an argument parser for command-line arguments
    parser = argparse.ArgumentParser()

    # Add arguments for epochs, batch sizes, and learning rate
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--train_batch_size",type=int,default=4)
    parser.add_argument("--valid_batch_size",type=int,default=2)
    parser.add_argument("--learning_rate",type=float,default=5e-5)

    # Parse the command-line arguments
    args = parser.parse_args()

    args.epochs
    args.train_batch_size



    # Set the device to CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Initialize the model
    model = DistilBERTClass()

    # Move the model to the selected device
    model.to(device)

    # Define the learning rate and optimizer
    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

    # Define the loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Train loop

    # Set the number of training epochs
    EPOCHS = 2

    # Start the training loop
    for epoch in range(EPOCHS):
        print(f"starting epoch: {epoch}")

        # Train the model for one epoch
        train(epoch, model, device, training_loader, optimizer, loss_function)

        # Validate the model
        valid(epoch, model, testing_loader, device, loss_function)


    # Get the SageMaker model directory from environment variables
    output_dir = os.environ['SM_MODEL_DIR']

    # Define the output path for the model file
    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')

    # Define the output path for the vocabulary file
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')

    # Save the model's state dictionary
    torch.save(model.state_dict(),output_model_file)

    # Save the tokenizer's vocabulary
    tokenizer.save_vocabulary(output_vocab_file)


# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
