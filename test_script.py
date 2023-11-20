import requests
from datasets import load_dataset
from main import app
# Load the first 100 records from SQuAD
squad_dataset = load_dataset("squad")['train'].select([i for i in range(100)])

# URL of your FastAPI service
api_url = "http://127.0.0.1:8000/qa"

for example in squad_dataset:
    question = example['question']
    context = example['context']

    # Send a request to the FastAPI service
    response = requests.post(api_url, json={"context": context, "question": question})

    # Print the results
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Actual Answer: {example['answers']['text'][0] if example['answers']['text'] else 'No answer'}")
    print(f"Generated Answer: {response.json()['answer']}")
    print("="*50)
