import streamlit as st
import requests
import numpy as np

# Vertex AI API 설정
vertex_ai_endpoint = "https://<vertex-ai-endpoint>"
access_token = "<your-access-token>"
index_endpoint_name = "<index-endpoint-name>"
deployed_index_id = "<deployed-index-id>"

# 임베딩 함수 (간단한 예시로 가정)
def embed_query(query):
    # 실제 임베딩 모델을 사용해 쿼리를 벡터로 변환해야 함
    return np.random.rand(100).tolist()

def search_vertex_ai(query):
    query_embedding = embed_query(query)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "index_endpoint": index_endpoint_name,
        "deployed_index_id": deployed_index_id,
        "queries": [{"embedding": query_embedding}],
        "num_neighbors": 10
    }
    
    response = requests.post(vertex_ai_endpoint, headers=headers, json=data)
    return response.json()

# Streamlit UI 구성
st.title("Vertex AI 벡터 검색")

query = st.text_input("검색어를 입력하세요:")

if st.button("검색"):
    if query:
        results = search_vertex_ai(query)
        st.write("검색 결과:")
        st.json(results)
    else:
        st.write("검색어를 입력하세요.")
