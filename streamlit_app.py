import streamlit as st
import requests

# Vertex AI 설정
vertex_ai_endpoint = "https://<vertex-ai-endpoint>"
embedding_model_endpoint = "https://asia-northeast3-aiplatform.googleapis.com/v1/projects/keyword-data-373606/locations/asia-northeast3/publishers/google/models/text-multilingual-embedding-002"
access_token = "ya29.a0AcM612w7j2vM8hgttkC9sSEl8sKTBaL-nFrqMTY3-V1zIdXwjwwYKNrYtDkwAl6BfgR80HbdE2SXcuX9g1g822lE-nF2cviTJLhr8Xa6YrFKscnM3u8vJsMSC1KDo-zlePmJEnTqiMPQMQx1fu2Btsl-Gd3MDUYPZtw7PORS9PcbsQaCgYKAXsSARESFQHGX2MiacXHD4Ns2vIB2oWGheil-g0181"
index_endpoint_name = "embedding_test"
deployed_index_id = "vertex_embedding_test_1724296756679"



def get_embedding_from_model(query):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "instances": [{"text": query}]
    }
    
    response = requests.post(embedding_model_endpoint, headers=headers, json=data)
    
    # 응답 상태 코드와 텍스트 출력
    st.write(f"Response Status Code: {response.status_code}")
    st.write(f"Response Text: {response.text}")
    
    # 응답이 성공적일 때만 JSON 파싱
    if response.status_code == 200:
        embedding = response.json()["predictions"][0]["embedding"]
        return embedding
    else:
        st.error("Failed to get embedding from model")
        return None

def search_vertex_ai(query):
    query_embedding = get_embedding_from_model(query)
    
    if query_embedding is None:
        st.error("Failed to retrieve embedding. Cannot proceed with search.")
        return {}
    
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
    
    # 응답 상태 코드와 텍스트 출력
    st.write(f"Response Status Code: {response.status_code}")
    st.write(f"Response Text: {response.text}")
    
    # 응답이 성공적일 때만 JSON 파싱
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to search in Vertex AI")
        return {}


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
