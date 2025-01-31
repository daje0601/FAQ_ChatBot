import re
import os
import pickle
import openai
import chromadb
from tqdm import tqdm
from typing import List
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv("./secret.env")

embedding_model_name = "text-embedding-3-small"
api_key = os.environ.get("OPENAI_API_KEY")

def clean_text(text: str) -> str:
    """FAQ 본문에서 불필요한 문자열(공백 등)을 제거하는 간단한 함수"""
    text = re.sub(r'\s+', ' ', text)  # 다중 공백을 단일 공백으로
    return text.strip()

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """FAQ 본문을 주어진 크기로 잘라 리스트로 반환"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    """
    1) final_result.pkl을 로드하여 (질문+답변) 텍스트를 전처리 및 청크 단위로 분할
    2) Chroma PersistentClient를 이용해 local DB(./embeddings/chroma_db)에 저장
    3) 대량 데이터를 처리하기 위해 batch_size 단위로 add
    """
    with open('./data/final_result.pkl', 'rb') as f:
        faq_data = pickle.load(f)
    
    # 1) ChromaDB 클라이언트 생성 (PersistentClient)
    client = chromadb.PersistentClient(path="./embeddings/chroma_db")
    
    # 2) faq_collection 생성 or 가져오기, 임베딩 함수 등록
    collection = client.get_or_create_collection(
        name="faq_collection",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, 
            model_name=embedding_model_name
        )
    )

    batch_size = 100
    all_chunks = []
    all_metadatas = []
    all_ids = []
    doc_id = 0
    
    # 3) 데이터 전처리 및 청크
    print("데이터 전처리 중...")
    for question, answer_text in faq_data.items():
        combined_text = clean_text(question + "\n\n" + answer_text)
        chunks = chunk_text(combined_text, chunk_size=500)
        
        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue
                
            all_chunks.append(chunk)
            all_metadatas.append({"source": question})
            all_ids.append(f"faq_{doc_id}")
            doc_id += 1
    
    # 4) 배치 처리 후 Chroma DB에 저장
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    for i in tqdm(range(total_batches), desc="벡터 DB 구축 중"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_chunks))
        
        batch_chunks = all_chunks[start_idx:end_idx]
        batch_metadatas = all_metadatas[start_idx:end_idx]
        batch_ids = all_ids[start_idx:end_idx]
        
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids
            # embeddings 인자를 주지 않는 이유:
            #   get_or_create_collection 시 등록한 embedding_function 이
            #   자동으로 documents를 임베딩해줍니다.
        )
    
    print(f"총 {doc_id}개의 청크가 DB에 저장되었습니다.")

if __name__ == "__main__":
    main()
