from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from Model import load_model, generate_answer
app = FastAPI()


class QARequest(BaseModel):
    context: str
    question: str


class DatasetRequest(BaseModel):
    data: list


@app.post("/qa")
async def qa_endpoint(request: QARequest):
    try:
        qa_model, tokenizer = load_model()
        answer = generate_answer(qa_model, tokenizer, request.context, request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dataset")
async def dataset_endpoint(request: DatasetRequest):
    try:
        data = request.data
        results = []

        for item in data:
            question = item['question']
            context = item['context']
            original_answer = item['answer'] if 'answer' in item else "No answer"

            qa_model, tokenizer = load_model()
            generated_answer = generate_answer(qa_model, tokenizer, context, question)

            results.append({
                'question': question,
                'original_answer': original_answer,
                'generated_answer': generated_answer
            })

        # Create a DataFrame and save it to a CSV file
        df = pd.DataFrame(results)
        df.to_csv("output.csv", index=False)

        return {"message": "CSV file created successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
