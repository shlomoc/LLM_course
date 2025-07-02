# News Category Classification with DistilBERT

This repository contains code for training, deploying, and testing a news headline classification model using DistilBERT on AWS SageMaker. The model classifies news headlines into four categories: Business, Science, Entertainment, and Health.

## Project Overview

This project demonstrates how to:
1. Fine-tune a pre-trained DistilBERT model for text classification
2. Deploy the model to AWS SageMaker
3. Create an AWS Lambda function to interact with the deployed model
4. Perform load testing on the deployed endpoint

## Repository Structure

- `script.py`: Main training script for the DistilBERT model
- `inference.py`: Inference script for model deployment
- `aws-lambda-llm-endpoint-invoke-function.py`: AWS Lambda function to invoke the SageMaker endpoint
- `TrainingNotebook.ipynb`: Jupyter notebook for training the model on SageMaker
- `SentimentAnalysis.ipynb`: Notebook for sentiment analysis tasks
- `EDA_MultiClassTextClassification.ipynb`: Exploratory data analysis for multi-class text classification
- `OptionalExperimentNotebook.ipynb`: Additional experiments with the model
- `Deployment.ipynb`: Notebook for model deployment
- `load-testing/`: Directory containing load testing scripts and input files

## Model Architecture

The model uses a pre-trained DistilBERT architecture with the following components:
- DistilBERT base model (uncased)
- A pre-classifier linear layer (768 -> 768)
- Dropout layer (0.3)
- Final classifier layer (768 -> 4) for the four news categories

## Dataset

The model is trained on a dataset of news headlines with four categories:
- Business
- Science
- Entertainment
- Health

The dataset is loaded from an S3 path and preprocessed before training.

## Training

To train the model:

1. Set up the necessary AWS permissions and SageMaker role
2. Update the S3 paths in `script.py` and `TrainingNotebook.ipynb`
3. Execute the training notebook to start the SageMaker training job

The training process includes:
- Data preprocessing and tokenization
- Model training with CrossEntropyLoss
- Validation during training
- Saving the model and tokenizer vocabulary

## Deployment

The model is deployed as a SageMaker endpoint using the `inference.py` script which handles:
- Loading the trained model
- Processing input requests
- Making predictions
- Formatting the responses

## Inference

The deployed model can be invoked via:
1. Direct SageMaker endpoint calls
2. The AWS Lambda function provided in `aws-lambda-llm-endpoint-invoke-function.py`

Input format:
```json
{
  "inputs": "Your news headline here"
}
```

Output format:
```json
{
  "predicted_label": "Category",
  "probabilities": [[p1, p2, p3, p4]]
}
```

## Load Testing

The `load-testing` directory contains scripts and input files for testing the performance of the deployed endpoint under load.

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- AWS SageMaker
- AWS Lambda (for serverless deployment)
- Boto3

## Getting Started

1. Clone this repository
2. Set up AWS credentials and permissions
3. Update the S3 paths in the code
4. Run the training notebook
5. Deploy the model using the deployment notebook
6. Test the endpoint with the provided scripts

## AWS Architecture

### How This Project Uses AWS Bedrock, API Gateway, Lambda, and S3

This project demonstrates a scalable machine learning application architecture using several AWS services working together:

#### AWS Bedrock Integration
- **Foundation Models**: The project leverages AWS Bedrock's foundation models for natural language processing tasks, providing powerful text classification capabilities without having to build models from scratch.
- **API Access**: AWS Bedrock provides a unified API to access various foundation models, which our Lambda function interfaces with for consistent inference.

#### API Gateway
- **RESTful API Endpoint**: API Gateway exposes our machine learning model as a RESTful API, allowing clients to make HTTP requests to classify news headlines.
- **Request Validation**: The API Gateway validates incoming requests, ensuring they contain the required parameters (like the 'headline' field in the JSON body).
- **Authentication & Authorization**: API Gateway handles authentication and authorization for API access, securing our model endpoints.

#### Lambda Functions
- **Serverless Inference**: The `aws-lambda-llm-endpoint-invoke-function.py` Lambda function processes incoming API requests, extracts the headline from the request body, and forwards it to the appropriate endpoint.
- **Payload Transformation**: The Lambda function transforms the API request into the format expected by the SageMaker/Bedrock endpoint and transforms the response back to a client-friendly format.
- **Cost Efficiency**: Lambda's serverless architecture ensures we only pay for actual usage, making the solution cost-effective for variable workloads.

#### S3 Integration
- **Data Storage**: Training data is stored in S3 (referenced in `script.py` via `s3_path = 'your-s3-uri-to-the-csv'`), providing a scalable and durable storage solution.
- **Model Artifacts**: Trained model artifacts are stored in S3, allowing for version control and easy deployment to different environments.
- **Deployment Pipeline**: S3 serves as a central repository in the CI/CD pipeline for model deployment.

#### Scalability Benefits
- **Horizontal Scaling**: API Gateway and Lambda automatically scale to handle varying loads without manual intervention.
- **Stateless Architecture**: The serverless components create a stateless architecture that can handle thousands of concurrent requests.
- **Regional Deployment**: The solution can be deployed across multiple AWS regions for global availability and reduced latency.

#### Security Considerations
- **IAM Roles**: Fine-grained access control through IAM roles ensures each component has only the permissions it needs.
- **API Keys**: API Gateway can use API keys to control access to the endpoints.
- **VPC Integration**: For enhanced security, Lambda functions can run within a VPC to access resources that aren't publicly accessible.

This architecture demonstrates how AWS services can be combined to create a production-ready, scalable machine learning application with minimal operational overhead.

## License

This project is provided as educational material for learning about NLP model deployment on AWS.
