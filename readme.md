# 스마트 스토어 FAQ 챗봇

**스마트스토어 FAQ** 데이터에 대해, 사용자의 질문을 벡터 임베딩으로 검색하고, LLM(GPT 계열)을 통해 **자동응답**을 제공하는 챗봇 프로젝트입니다.

---

## 개요

- **Chroma**(로컬 벡터 DB)를 이용해 FAQ 임베딩을 저장하고 검색합니다.
- **RAG 구조**를 사용해 FAQ 데이터에서 유사한 청크를 찾아 **ChatGPT**로 전달합니다.
- **FastAPI** 서버에서 `/chat` 엔드포인트를 통해 **Streaming** 형태의 답변을 반환합니다.

---

## 폴더 구조

```bash
├── app.py                  # FastAPI 서버 (FAQ 검색/ChatGPT 스트리밍)
├── build_index.py          # FAQ 데이터를 분할(청크)하고 벡터 DB에 저장하는 스크립트
├── data
│   └── final_result.pkl    # 최종 FAQ 자료 (질문-답변 쌍)
├── data_check.ipynb        # FAQ 데이터 탐색/점검 노트북
├── embeddings
│   ├── chroma_db_1000
│   │   ├── ...
│   │   └── chroma.sqlite3  # 1000토큰 chunk 기반 벡터 DB
│   └── chroma_db_500
│       ├── ...
│       └── chroma.sqlite3  # 500토큰 chunk 기반 벡터 DB
├── readme.md               # 프로젝트 소개 및 사용법 (이 파일)
├── requirements.txt        # Python 의존성 목록
└── secret.env              # OpenAI API Key 등 비밀 환경변수
```

---

## 사전 준비

1. **Python 3.11+** 환경이 필요합니다.  
2. **OpenAI API Key**를 발급받아, 프로젝트 루트의 `secret.env`에 다음과 같이 저장합니다:
   ```bash
   OPENAI_API_KEY="sk-..."
   ```
3. 기타 임베딩 모델/LLM 설정은 `app.py` 내부의 모델(`model_name`, `embedding_model_name`)은 수정 가능합니다.

---

## 설치 및 실행 방법

### 1) 라이브러리 설치
```bash
pip install -r requirements.txt
```


### 2) FAQ 인덱스 구축 (선택)
- 이미 `embeddings/chroma_db_1000` 폴더 내에 벡터DB가 완비된 경우 생략 가능  
- 새로 FAQ 데이터를 임베딩하고 싶다면, 아래처럼 **청크 크기**를 선택해 인덱스를 구축합니다:
  ```bash
  python build_index.py --chunk-size 1000
  ```
  - `--chunk-size` 기본값은 500, 필요 시 1000 등으로 조절 가능  
  - 실행 후, `embeddings/chroma_db_1000` 폴더에 새 DB가 생성됩니다.

### 3) FastAPI 서버 실행
```bash
python app.py
```
- 기본 포트 `8000`에서 구동됩니다.  
- `0.0.0.0:8000/chat` 엔드포인트에 **POST** 요청을 보내면 응답을 스트리밍합니다.

---

## 사용 예시
```bash 
# 1. env 만들기 
conda create -n coxwave python=3.11 -y 
conda activate coxwave 

# 2. secret.env 만들기 
1) 터미널에 vi secret.env 입력하기 
2) OPENAI_API_KEY = "your_openai_key" 를 입력합니다. 
3) 키보드 ESC를 누릅니다. 
4) :wq 를 입력합니다. 

# 3. requirements 설치하기 
pip install -r requirements.txt 

# 3-1. 만약 chromaDB를 만들고 싶다면, 실행합니다 
python build_index.py --chunk-size 1000

# 4. app 파일 실행 
python app.py 

# 5. 질문하기 - 다른 터미널을 열고 아래 명령어를 입력하세요 
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{"session_id": "test123", "question": "반품 접수된 주문을 교환으로 변경하고 싶어요."}'
```

---

## 주요 기능

1. **build_index.py**  
   - `final_result.pkl` 내 **질문+답변** 텍스트를 전처리 후, 설정한 **청크 크기**만큼 분할  
   - `chroma_client`(PersistentClient)로 벡터 DB에 저장  
   - **배치(batch_size=100)** 단위로 임베딩 추가 → 대량 데이터 처리 시 속도/안정성 확보

2. **app.py**  
   - **FastAPI** 서버 구동  
   - `/chat` 엔드포인트로 질문을 받아,  
     1. **블록 키워드** 필터 → Out-of-Domain 처리  
     2. **Chroma**를 이용해 **FAQ 검색**  
     3. 검색 결과가 없으면 → Out-of-Domain  
     4. 있으면 → **ChatCompletion** API로 답변  
     5. 스트리밍(`yield`)으로 결과 반환  

3. **데이터**  
   - `data/final_result.pkl` 에 FAQ가 (질문:답변) 딕셔너리 형태로 저장  
   - `data_check.ipynb`에서 데이터 분포 분석 및 샘플링 확인 가능

---


