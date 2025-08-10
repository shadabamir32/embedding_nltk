import streamlit as st
import requests
import pandas as pd
from enums.embeddingmodels import EmbeddingModels

st.title("Embeddings")

FASTAPI_URL = "http://localhost:8000"

# Init session state
if "file_hash" not in st.session_state:
    st.session_state["file_hash"] = None
# if "training_completed" not in st.session_state:
#     st.session_state["training_completed"] = False
# if "selected_model" not in st.session_state: 
#     st.session_state["selected_model"] = None
models = [model.value for model in EmbeddingModels]

# File upload
st.subheader("Upload a dataset")
uploaded_file = st.file_uploader("Choose a file", type=["json"])
if uploaded_file is not None and st.session_state.file_hash is None:
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(f"{FASTAPI_URL}/api/v1/upload", files=files)
    if response.status_code == 200:
        file_hash = response.json().get("data").get("file_hash")
        st.session_state.file_hash = file_hash
        st.toast("File uploaded successfully!", icon="✅")
    else:
        st.error(f"Failed to upload file. {response.json().get('message')} ")
# Display file hash
st.info(f"File hash: {st.session_state.file_hash if st.session_state.file_hash else 'NA'}")

# Model training
if st.session_state.file_hash is not None:
    st.subheader("Train a Model")
    selected_model = st.selectbox("Select Model", models)
    # if st.session_state.selected_model != selected_model:
    #     st.session_state.selected_model = None
    #     st.session_state.training_completed = False
    if selected_model and st.button("Train Model"):
        payload = {"file_hash": st.session_state.file_hash}
        print(f"Training model: {selected_model} with file hash: {st.session_state.file_hash}")
        response = requests.post(f"{FASTAPI_URL}/api/v1/train/{selected_model}", data=payload)
        if response.status_code == 200:
            # st.session_state.selected_model = selected_model
            # st.session_state.training_completed = True
            st.toast(f"{response.json().get('message')}", icon="✅")
        else:
            # st.session_state.selected_model = None
            # st.session_state.training_completed = False
            st.error(f"Failed to start training. {response.json()}")
    #st.info(f"Selected trained Model: {st.session_state.selected_model}")

# Display selected model
# if st.session_state.file_hash is not None and st.session_state.selected_model is not None and st.session_state.training_completed is True:
    st.subheader("Lookup Similar Words")
    query = st.text_input("Enter a word to search for similar words")
    search_selected_model = st.selectbox("Select Model for Similarity Search", models)
    if search_selected_model and st.button("Search") and query:
        params = {
            "file_hash": st.session_state.file_hash,
            "query": query
        }
        url = f"{FASTAPI_URL}/api/v1/lookup/{search_selected_model}"
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get("data", [])
            if results:
                df = pd.DataFrame(results)
                if search_selected_model == EmbeddingModels.tfidf.value:
                    df.rename(columns={
                        "word": "Word",
                        "similarity": "Similarity",
                        "doc_index": "Document Index"
                    }, inplace=True)
                else:
                    df.rename(columns={
                        "word": "Word",
                        "similarity": "Similarity"
                    }, inplace=True)
                # Format similarity column
                df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No similar words found.")
        else:
            st.error(f"Lookup failed: {response.json().get('message', response.text)}")
