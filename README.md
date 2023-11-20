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

## APIs

1. Question Answering Endpoint (/qa)

- **Input Format:**

  - HTTP Method: POST

  - Endpoint: /qa

  - Request Body: JSON with the following structure:

   ```json
   {
     "context": "Your context paragraph goes here.",
     "question": "Your question goes here."
   }
   ```

    

- **Output Format:**

  - JSON response with the generated answer:

   ```json
   {
     "answer": "Generated answer for the given question."
   }
   ```

    


2. Dataset Processing Endpoint (/dataset)

  - **Input Format**:

    - HTTP Method: `POST`

    - Endpoint: `/dataset`

    - Request Body: JSON with the following structure:

     ```
     {
       "data": [
         {
           "context": "Context for the first question.",
           "question": "First question.",
           "answer": "Original answer for the first question."
         }
       ]
     }
     ```

      

    **Output Format**:

    - JSON response indicating the success of CSV creation:

    ```json
    {
      "message": "CSV file created successfully!"
    }
    
    ```

    



# Model Explanation

the model used is BERT large model (uncased) whole word masking finetuned on SQuAD
This model should be used as a question-answering model. You may use it in a question answering pipeline, or use it to output raw results given a query and a context.

## Whole Word Masking Approach

Differently to other BERT models, this model was trained with a new technique: Whole Word Masking. In this case, all of the tokens corresponding to a word are masked at once. The overall masking rate remains the same.
