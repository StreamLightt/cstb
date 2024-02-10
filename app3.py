import streamlit as st
from huggingface_hub import InferenceClient
import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
API_TOKEN = st.secrets["API_TOKEN"]

headers = {"Authorization": "Bearer {API_TOKEN}"}  
API_URL = "https://api-inference.huggingface.co/models/"
df = pd.read_excel('chapes-fluides.xlsx')
inference_client = InferenceClient(token=API_TOKEN)

# Function to vectorize text - assuming this is already defined in your code
def create_index(data, text_column, model):
    # Encode the text column to generate embeddings
    embeddings = model.encode(data[text_column].tolist())

    # Dimension of embeddings
    dimension = embeddings.shape[1]

    # Prepare the embeddings and their IDs for FAISS
    db_vectors = embeddings.astype(np.float32)
    db_ids = np.arange(len(data)).astype(np.int64)

    # Normalize the embeddings
    faiss.normalize_L2(db_vectors)

    # Create and configure the FAISS index
    index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(db_vectors, db_ids)

    return index, embeddings

#Function to vectorize txt, use model.encode
def vectorize_text(model, text):
    # Encode the question to generate its embedding
    question_embedding = model.encode([text])

    # Convert to float32 for compatibility with FAISS
    question_embedding = question_embedding.astype(np.float32)

    # Normalize the embedding
    faiss.normalize_L2(question_embedding)

    return question_embedding

def extract_context(indices, df,i):
    # Extracting only the first index
    index_i = indices[0][i]
    context = df.iloc[index_i]['text_segment']
    return context

def generate_answer_from_context(context, client, model,prompt):
    try:
        # Use a hypothetical text generation method if available
        answer = client.text_generation(prompt=prompt, model=model, max_new_tokens=250)

        answer_cleaned = re.sub(r'^.*Answer:', '', answer).strip()
        return answer_cleaned
    except Exception as e:
        print(f"Error encountered: {e}")
        return None


# Load model    
model_sentence_transformers = SentenceTransformer('intfloat/multilingual-e5-base')
model_reponse_mixtral_instruct="mistralai/Mixtral-8x7B-Instruct-v0.1"

#Load the index
index_reloaded = faiss.read_index("./index/chapes_fluides_e5.index")

K=2

# Streamlit app interface
st.title("CSTB App")

if "messages" not in st.session_state:
    st.session_state.messages = []

if user_question := st.chat_input("Votre question : "):
    # Vectorize the user question and search in the FAISS index
    st.session_state.messages.append({"role": "user", "content": user_question})
    question_embedding = vectorize_text(model_sentence_transformers, user_question)
    D, I = index_reloaded.search(question_embedding, K)  # question_embedding is already 2D

    # Extract context for the top K results
    context = extract_context(I, df, 0) + ' ' + extract_context(I, df, 1)
    prompts = [
        f"Répondre à cette question : {user_question} en utilisant le contexte suivant {context}. Etre le plus précis possible et ne pas faire de phrase qui ne se finit pas \nReponse:"
        #Autre prompt possible 
        #f"Contexte: {context}\nQuestion: {user_question}\nReponse:", 

    ]

    # Generate answers using different prompts
    answers = [generate_answer_from_context(context, inference_client, model_reponse_mixtral_instruct,prompts[i]) for i in range(len(prompts))]
    # Display answers
    for i, answer in enumerate(answers):
        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            #st.markdown(answer)
            #st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Failed to generate an answer."})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])