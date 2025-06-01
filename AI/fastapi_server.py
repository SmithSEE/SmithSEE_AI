# fastapi_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

model = BertForSequenceClassification.from_pretrained("sseul2/bert-smishing-model2")
tokenizer = BertTokenizer.from_pretrained("sseul2/bert-smishing-model2")

class InputText(BaseModel): 
    text: str

@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["ham", "smishing"]
    result = {labels[i]: float(probs[0][i]) for i in range(2)}
    return result
