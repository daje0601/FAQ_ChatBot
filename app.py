# 파일명: app.py
import os
import openai
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict

import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./secret.env")

model_name = "gpt-4o-mini"
embedding_model_name = "text-embedding-3-small"
api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
  api_key=api_key
)

app = FastAPI()

# -------------------------
# 0) 전역 설정: Chroma Client & Collection
# -------------------------
# 1) PersistentClient 로 영구저장 방식 지정
chroma_client = PersistentClient(path="./embeddings/chroma_db")

# 2) OpenAI EmbeddingFunction 등록
openai_embedding = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name=embedding_model_name
)

# 3) Collection 가져오기 (임베딩 함수 등록)
collection = chroma_client.get_collection(
    name="faq_collection",
    embedding_function=openai_embedding
)

# -------------------------
# 간단한 메모리: 세션별 대화 히스토리
# 실제 운영에서는 Redis/DB 등으로 교체 가능
# -------------------------
SESSION_HISTORY: Dict[str, List[Dict[str, str]]] = {}


# -------------------------
# 헬퍼 함수
# -------------------------
def retrieve_faq_chunks(query: str, top_k: int = 5, threshold: float = 0.8) -> List[str]:
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['distances','embeddings', 'documents', 'metadatas'],
        )
        
        # 결과가 없는 경우 처리
        if not results["documents"] or len(results["documents"][0]) == 0:
            return []

        docs_list = results["documents"][0]
        dist_list = results["distances"][0]

        filtered_docs = []
        for doc, dist in zip(docs_list, dist_list):
            # distance가 낮을수록 더 유사(Chroma 기본=euclidean)
            if dist < threshold:
                filtered_docs.append(doc)

        return filtered_docs
        
    except Exception as e:
        print(f"검색 중 오류 발생: {str(e)}")
        return []

def format_history(history: List[Dict[str, str]]) -> str:
    """
    이전 대화 내용을 "USER: ..., ASSISTANT: ..." 형태로 이어붙여 문자열로 반환
    """
    formatted = ""
    for turn in history:
        role = turn["role"].upper()
        content = turn["content"]
        formatted += f"{role}: {content}\n"
    return formatted

def openai_stream_chat(messages, model=model_name):
    """
    ChatCompletion API (stream=True) 사용, 토큰 단위로 yield하여 스트리밍 응답
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=800,    # 답변 길이 여유
        top_p=1.0,
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


# -------------------------
# FastAPI 라우터: /chat
# -------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    예시 request body:
    {
      "session_id": "abc123",
      "question": "스마트스토어의 미성년자 회원가입 관련 질문"
    }
    """
    data = await request.json()
    session_id = data.get("session_id", "default")
    user_question = data.get("question", "")

    # (1) 스마트스토어 관련 없는 질문 필터 (간단 예시)
    blocked_keywords = ["맛집", "날씨", "영화", "환율", "주식"]
    if any(bk in user_question for bk in blocked_keywords):
        # out-of-domain
        system_msg = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
        follow_up = " - 음식도 스토어 등록이 가능한지 궁금하신가요?"

        def stream_out_of_domain():
            yield system_msg
            yield "\n"
            yield follow_up

        return StreamingResponse(stream_out_of_domain(), media_type="text/plain")

    # (2) 세션 히스토리 가져오기
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []

    conversation_history = SESSION_HISTORY[session_id]

    # (3) FAQ chunk 검색
    relevant_docs = retrieve_faq_chunks(user_question, top_k=3)

    # (4) 시스템 프롬프트 구성
    system_prompt = """당신은 네이버 스마트스토어 FAQ 안내를 돕는 어시스턴트입니다.

주의사항:
1) 아래에 제공된 FAQ 데이터(질문/답변)을 최우선 근거로 사용합니다.
2) 이전 대화(히스토리) 중 현재 질문과 직접적으로 관련 없는 내용이 섞여 있다면, 절대 답변에 포함하지 마세요.
3) 이전 대화(히스토리)는 현재 질문과 연관이 있을 때에만 참고하세요.
4) 만약 FAQ 데이터에 내용이 없다면 "FAQ에 해당 내용이 없습니다"라고 안내하세요.
5) 스마트스토어와 전혀 무관한 질문이라면 "스마트 스토어 FAQ를 위한 챗봇..." 식으로 안내해주세요.
6) 답변 후, 사용자가 궁금해할 만한 후속 질문을 1~2개 제안해주세요.
"""
    system_context = "\n\nFAQ 자료(요약):\n" + "\n\n".join(relevant_docs)

    # (5) 이전 대화 + 이번 질문 합쳐서 user 메시지 구성
    conversation_text = format_history(conversation_history)
    user_combined_msg = conversation_text + "\n\n" + user_question

    messages = [
        {"role": "system", "content": system_prompt + system_context},
        {"role": "user", "content": user_combined_msg}
    ]

    # (6) 스트리밍 생성기
    def stream_chat_response():
        for token in openai_stream_chat(messages):
            yield token

    # (7) 사용자 대화 히스토리 갱신
    conversation_history.append({"role": "user", "content": user_question})

    return StreamingResponse(stream_chat_response(), media_type="text/plain")


# -------------------------
# uvicorn 실행 (개발용)
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
