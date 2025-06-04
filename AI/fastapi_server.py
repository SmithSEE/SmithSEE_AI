# uvicorn AI.fastapi_server:app --host 0.0.0.0 --port 8001

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# 모델 로딩
tokenizer = BertTokenizer.from_pretrained("sseul2/bert-smishing-model-final")
model = BertForSequenceClassification.from_pretrained("sseul2/bert-smishing-model-final")

# 입력 모델
class InputText(BaseModel):
    text: str

# 응답 모델
class PredictionResult(BaseModel):
    smishing: str
    riskScore: float

# [post] /predict 텍스트 받아와서 예측 
@app.post("/predict", response_model=PredictionResult)
def predict(input: InputText):
    print("받은 텍스트:", input.text)

    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    smishing_score = float(probs[0][1])
    is_smishing = smishing_score > 0.5

    return {
        "smishing": "LABEL_1" if is_smishing else "LABEL_0",
        "riskScore": round(smishing_score, 4)
    }


# 들어온 텍스트 전처리


# 마지막에 텍스트 모델에 학습
