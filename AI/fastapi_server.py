# uvicorn fastapi_server:app --host 0.0.0.0 --port 8001

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI()

# 모델 로딩
model = BertForSequenceClassification.from_pretrained("sseul2/bert-smishing-model2")
tokenizer = BertTokenizer.from_pretrained("sseul2/bert-smishing-model2")

# 입력 모델
class InputText(BaseModel):
    text: str

# 응답 모델
class PredictionResult(BaseModel):
    smishing: str
    riskScore: float

@app.post("/predict", response_model=PredictionResult)
def predict(input: InputText):
    print("📥 받은 텍스트:", input.text)

    # 토크나이징 및 예측
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    smishing_score = float(probs[0][1])
    is_smishing = smishing_score > 0.5

    # 출력 로그
    print(f"모델 예측 확률 - 정상: {probs[0][0]:.4f}, 스미싱: {probs[0][1]:.4f}")
    print(f"최종 결과 - smishing: {'LABEL_1' if is_smishing else 'LABEL_0'}, riskScore: {smishing_score:.4f}")

    return {
        "smishing": "LABEL_1" if is_smishing else "LABEL_0",
        "riskScore": round(smishing_score, 4)
    }
