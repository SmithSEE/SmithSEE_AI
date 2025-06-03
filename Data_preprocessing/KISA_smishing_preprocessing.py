
# !pip install git+https://github.com/ssut/py-hanspell.git
# !pip install git+https://github.com/haven-jeon/PyKoSpacing.git

import pandas as pd
import re
from pykospacing import Spacing
from hanspell import spell_checker
from tqdm import tqdm
import csv

tqdm.pandas()
spacing = Spacing()

# 데이터 불러오기
df = pd.read_csv("KISA_smishing_data.csv")

# 맞춤법 교정 함수
def safe_spell_check(text):
    try:
        if isinstance(text, str) and 1 < len(text) <= 500:
            result = spell_checker.check(text)
            if hasattr(result, 'checked'):
                return result.checked
    except:
        pass
    return text

# 전처리 함수
def preprocess_text(text):
    if not isinstance(text, str):
        return text

    text = text.replace("ifg@", "")

    # URL 치환: http/https/www 또는 축약 주소 포함
    url_pattern = r"(https?://|http?://|www\.)[^\s]+|\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?"
    text = re.sub(url_pattern, "URL", text)

    # 날짜 마스킹
    text = re.sub(r"\d{1,2}\.\d{1,2}\s*[\-~]\s*\d{1,2}\.\d{1,2}(일)?", "DATE", text)
    text = re.sub(r"\d{4}년\s?\d{1,2}월\s?\d{1,2}일", "DATE", text)
    text = re.sub(r"\d{1,2}월\s?\d{1,2}일?", "DATE", text)
    text = re.sub(r"\d{1,4}년\s?\d{1,2}월", "DATE", text)
    text = re.sub(r"\d{1,2}/\d{1,2}\(.\)", "DATE", text)
    text = re.sub(r"\d{1,2}/\d{1,2}\(\w+\)", "DATE", text)
    text = re.sub(r"\d{1,2}일", "DATE", text)
    text = re.sub(r"\d{1,4}년", "DATE", text)
    text = re.sub(r"\d{1,2}월", "DATE", text)

    # 시간 마스킹
    text = re.sub(r"\d{2}:\d{2}", "TIME", text)
    text = re.sub(r"\d{1,2}시간", "TIME", text)
    text = re.sub(r"\d{1,2}시", "TIME", text)
    text = re.sub(r"\d{1,2}시\s*[\-~]\s*\d{1,2}시", "TIME", text)

    # 로또 마스킹
    text = re.sub(r"L\*TT\*", "로또", text)

    # 공백 압축
    text = re.sub(r"\s{2,}", " ", text)

    # 숫자/전화번호/마스킹 → NUM
    text = re.sub(r"[\*\d,\-_\s]{6,}", "NUM", text)
    text = re.sub(r"\d+\.\d+", "NUM", text)  # 99.9 → NUM 소숫점 자리
    text = re.sub(r"\d+", "NUM", text)


    # 기관 키워드 복원
    text = re.sub(r"\b(정부|민원|교통|경찰청|경찰)NUM\b", r"\g<1>24", text)

    # 특수문자 제거
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

    # 띄어쓰기
    text = spacing(text)

    # 맞춤법
    text = safe_spell_check(text)

    return text

# 전처리 적용
df["clean_df"] = df["SMS"].progress_apply(preprocess_text)

# 중복 제거
df = df.drop_duplicates(subset=["clean_df"]).copy()

# 저장
df[["SMS","clean_df"]].to_csv("KISA_smishing_clean_test.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
