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
