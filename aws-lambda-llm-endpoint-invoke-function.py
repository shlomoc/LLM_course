# Import necessary libraries
import json
import boto3

# Define the main Lambda handler function
def lambda_handler(event, context):

    # Create a SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Parse the request body from the Lambda event
    body = json.loads(event['body'])

    # Extract the headline from the request body
    headline = body['query']['headline']

    #headline = "How I met your Mother voted as best sitcom in Europe"

    # Specify the name of the SageMaker endpoint
    endpoint_name = 'your-endpoint-name'

    # Create the payload for the endpoint invocation
    payload = json.dumps({"inputs": headline})

    # Invoke the SageMaker endpoint with the payload
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType = "application/json",
        Body = payload
        )
    # Read and decode the response from the endpoint
    result = json.loads(response['Body'].read().decode())

    # Return the result with a 200 status code
    return {
        'statusCode':200,
        'body': json.dumps(result)
    }
