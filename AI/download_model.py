from transformers import BertTokenizer, BertForSequenceClassification

# 모델 저장할 디렉토리
save_path = "./model/bert-smishing-model"

# 모델 다운로드 (처음 한 번만 실행)
tokenizer = BertTokenizer.from_pretrained("sseul2/bert-smishing-model")
model = BertForSequenceClassification.from_pretrained("sseul2/bert-smishing-model")

# 저장
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
