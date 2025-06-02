from transformers import BertTokenizer, BertForSequenceClassification

# 모델 캐시 경로 설정
cache_dir = "./model_cache"

# 다운로드 (한 번만 실행하면 됨)
tokenizer = BertTokenizer.from_pretrained("sseul2/bert-smishing-model", cache_dir=cache_dir)
model = BertForSequenceClassification.from_pretrained("sseul2/bert-smishing-model", cache_dir=cache_dir)

print("모델 캐시 완료")
