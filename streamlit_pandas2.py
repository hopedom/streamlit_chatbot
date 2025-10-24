import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# RunnablePassthrough, ChatPromptTemplate, StrOutputParser는 그대로 사용
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

# ----------------------------------------------------
# 0. 환경 설정 (API 키 설정)
# ----------------------------------------------------
load_dotenv()
try:
    # st.secrets에서 키를 먼저 찾고, 없으면 .env에서 찾습니다.
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
         raise ValueError("OpenAI API 키를 찾을 수 없습니다.")
    os.environ["OPENAI_API_KEY"] = api_key
except Exception as e:
    st.error(f"API 키 로드 오류: {e}. .streamlit/secrets.toml 또는 .env 파일을 확인하세요.")
    st.stop()
    
# ----------------------------------------------------
# 1. 상수 및 유틸리티 함수
# ----------------------------------------------------
CHROMA_PATH = './chromadb/pandas_rst'
EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = "gpt-4o-mini"

def format_docs(docs: List[Document]) -> str:
    """검색된 LangChain Document 객체들을 하나의 문자열 컨텍스트로 결합"""
    return "\n\n".join(doc.page_content for doc in docs)

def safe_filename(path: str) -> str:
    """경로에서 파일명만 안전하게 추출"""
    try:
        return os.path.basename(path)
    except Exception:
        return "unknown"

def map_source_to_label(source: str) -> str:
    """source가 URL이면 클릭 가능한 링크로, 아니면 파일명만 표시"""
    if not source:
        return "`unknown`"
    if source.startswith("http"):
        filename = safe_filename(source)
        return f"[{filename}]({source})"
    # 로컬 경로일 경우
    filename = safe_filename(source)
    return f"`{filename}`"

# ----------------------------------------------------
# 2. 캐싱을 통한 RAG 컴포넌트 로드/생성
# ----------------------------------------------------
@st.cache_resource
def get_vector_store():
    """ChromaDB를 로드하고 OpenAI 임베딩 함수를 사용하여 벡터 저장소를 준비"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL)
        )
        # ChromaDB 컬렉션이 비어있는지 간단히 검증
        if vectorstore._collection.count() == 0:
             st.error("ChromaDB 컬렉션이 비어있습니다. 임베딩이 필요합니다. 'python 쳇봇_v0.ipynb'를 실행했는지 확인하세요.")
             return None
             
        return vectorstore
    except Exception as e:
        st.error(f"ChromaDB 로드 중 심각한 오류 발생. 원인: {type(e).__name__}: {e}")
        return None

def setup_rag_components(_vectorstore: Chroma):
    """LLM, 프롬프트, Retriever 설정 및 RAG 체인 구축"""
    if _vectorstore is None:
        return None

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # 답변 생성 프롬프트
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks focused on Pandas documentation. "
         "Use the following pieces of retrieved context to answer the question. "
         "If you don't know the answer, just say that you don't know. "
         "Use three sentences maximum and keep the answer concise.\n\n"
         "Context: {context}"),
        ("human", "{question}")
    ])

    # 1. 문서 검색 체인: 검색 결과를 format_docs로 문자열화
    context_chain = retriever | format_docs
    
    # 2. RAG 체인: 검색된 컨텍스트와 질문을 LLM에 전달하여 답변 생성
    # RunnableParallel을 사용하여 검색 결과와 답변을 모두 반환하도록 체인 구조를 변경합니다.
    # 'retrieved_docs' 키를 통해 원본 문서 리스트를 별도로 저장하여 출처 표시에 사용합니다.
    rag_chain = RunnableParallel(
        # 'answer' 키에 답변을 저장 (컨텍스트 문자열 + 질문 -> LLM -> 문자열)
        answer = {
            "context": context_chain, 
            "question": RunnablePassthrough()
        } | qa_prompt | llm | StrOutputParser(),
        
        # 'retrieved_docs' 키에 검색된 문서 객체 리스트를 그대로 저장 (출처 표시용)
        retrieved_docs = retriever
        
    )
    
    return rag_chain

# ----------------------------------------------------
# 3. Streamlit UI 정의
# ----------------------------------------------------
st.set_page_config(page_title="Pandas RAG Q&A 🐼", layout="wide")
st.title("🐼 Pandas 공식 문서 기반 RAG 챗봇")
st.markdown("---")

vectorstore = get_vector_store()
rag_chain = setup_rag_components(vectorstore)


if rag_chain:
    user_question = st.text_input(
        "Pandas에 대해 궁금한 점을 질문하세요:",
        placeholder="예: DataFrame에서 누락된 값(NaN)을 확인하는 메서드는 무엇인가요?"
    )

    if user_question:
        with st.spinner("Pandas 문서에서 답변을 검색하는 중..."):
            try:
                # 1. RAG 체인 실행 (이제 'answer'와 'retrieved_docs' 딕셔너리를 반환)
                response = rag_chain.invoke(user_question)

                # 2. 답변 표시
                st.subheader("🤖 답변")
                st.info(response['answer'])

                # 3. 검색된 출처 표시 (오류 없이 복구)
                st.subheader("🔍 검색된 출처 (Context)")
                retrieved_docs = response['retrieved_docs'] # 체인 결과에서 문서 리스트를 가져옴

                for i, doc in enumerate(retrieved_docs):
                    source_label = map_source_to_label(doc.metadata.get("source", ""))
                    with st.expander(f"청크 {i+1} • 출처: {source_label}"):
                        # doc.page_content는 문서 청크 내용
                        st.code(doc.page_content, language="text")

            except Exception as e:
                st.error(f"RAG 체인 실행 중 오류 발생: {type(e).__name__}: {e}")
else:
    # vectorstore 또는 rag_chain이 None인 경우 (API 키 또는 ChromaDB 경로 문제)
    st.warning("RAG 시스템을 초기화하지 못했습니다. 상단의 오류 메시지나 터미널을 확인하세요.")