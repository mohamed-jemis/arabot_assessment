# Arabot
### Question-Answering Coding Exercise

This web service is built using FastAPI and utilizes the Hugging Face Transformers library for Question Answering which enables various model for question answering without needd to pretrain your own model.

# Clone the repository:
```
git clone <https://mohamed-jemis/arabot_assessment>
cd <arabot>
```

# Install project dependencies using Poetry:

```
poetry install
```

# Usage
### Running the Web Service
To start the FastAPI web service, run the following command within your project directory:
```
poetry run uvicorn main:app --reload
```
The service will start on http://localhost:8000 by default.


# Model Explanation
the model used is BERT large model (uncased) whole word masking finetuned on SQuAD
This model should be used as a question-answering model. You may use it in a question answering pipeline, or use it to output raw results given a query and a context.

## Whole Word Masking Approach
Differently to other BERT models, this model was trained with a new technique: Whole Word Masking. In this case, all of the tokens corresponding to a word are masked at once. The overall masking rate remains the same.
