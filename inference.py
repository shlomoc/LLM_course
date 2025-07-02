# Import necessary libraries
import torch
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel

# Define the maximum length for tokenized sequences
MAX_LEN = 512

# Define the DistilBERT model class for classification (must match the training script)
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


# Function to load the model from the model directory (SageMaker specific)
def model_fn(model_dir):

    print("Loading model from :", model_dir)

    # Initialize the model architecture
    model = DistilBERTClass()
    # Load the saved model state dictionary
    model_state_dict = torch.load(os.path.join(model_dir, 'pytorch_distilbert_news.bin'),map_location = torch.device('cpu'))
    model.load_state_dict(model_state_dict)


    return model

# Function to process the input request (SageMaker specific)
def input_fn(request_body,request_content_type):

    # Check if the content type is JSON
    if request_content_type == 'application/json':
        # Parse the JSON request body and extract the input sentence
        input_data = json.loads(request_body)
        sentence = input_data['inputs']
        return sentence
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# Function to make predictions on the input data (SageMaker specific)
def predict_fn(input_data, model):

    # Set the device and move the model to it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Tokenize the input data and move tensors to the device
    inputs = tokenizer(input_data, return_tensors="pt").to(device)

    # Extract input IDs and attention mask
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference without calculating gradients
    with torch.no_grad():
        outputs = model(ids,mask)

    # Calculate probabilities using softmax and convert to a NumPy array
    probabilities = torch.softmax(outputs, dim = 1).cpu().numpy()

    # Define class names and determine the predicted label
    class_names = ["Business", "Science", "Entertainment", "Health"]
    predicted_class = probabilities.argmax(axis=1)[0]
    predicted_label = class_names[predicted_class]


    # Return the prediction results as a dictionary
    return {'predicted_label': predicted_label, 'probabilities':probabilities.tolist()}



# Function to format the prediction output (SageMaker specific)
def output_fn(prediction, accept):

    # If the accept header is JSON, serialize the prediction to a JSON string
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
