from datasets import load_dataset
from transformers import BertTokenizer, TFBertForQuestionAnswering, pipeline
import tensorflow as tf


def load_model():
    """
    Loading the pre trained model for question answering from the hugging face transformers library
    """
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = pipeline('question-answering', model=model_name)
    return model, tokenizer


def generate_answer(model, model_tokenizer, _context, _question):
    """
    Generating an answer given the context and the question

    Arguments :
    model-> provide the pretrained model
    model_tokenizer -> provide the tokenizer loaded
    _context -> the context needed for the generation
    _question -> provide the question needed to be fed to the model
    """
    inputs = model_tokenizer(_question, _context, return_tensors="tf")
    outputs = model(inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = tf.argmax(start_scores, axis=1).numpy()[0]
    answer_end = tf.argmax(end_scores, axis=1).numpy()[0] + 1

    generated_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0][answer_start:answer_end]))
    return generated_answer

# Testing the Model
if __name__ == "__main__":
    # Download the SQuAD dataset
    squad_dataset = load_dataset("squad")

    # Load only the first 100 records
    squad_dataset = squad_dataset['train'].select([i for i in range(100)])

    # Load the DistilBERT QA model
    qa_model, tokenizer = load_model()

    # testing the model on the first 100 records
    correct_predictions = 0
    total_predictions = 0

    for example in squad_dataset:
        question = example['question']
        context = example['context']
        answer = example['answers']['text'][0] if example['answers']['text'] else "No answer"

        prediction = qa_model(question=question, context=context)
        predicted_answer = prediction['answer']

        print("Question:", question)
        print("Actual Answer:", answer)
        print("Predicted Answer:", predicted_answer)
        print("="*50)

        total_predictions += 1
        if predicted_answer == answer:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")
