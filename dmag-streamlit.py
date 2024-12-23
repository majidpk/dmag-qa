import streamlit as st
import time

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List
from langchain_core.runnables import RunnablePassthrough

# Function to initialize the QA chain and retriever
@st.cache_resource
def initialize_qa_chain():
    persist_directory = "./vectordb"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    # Set up the retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Initialize OpenAI LLM
    avalai_api = "aa-Qtp7NPuYMqOGnAWUwZR5rxruRPGrYohBbRBhJZC3srcWW7Xc"
    openai_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", base_url="https://api.avalai.ir/v1", api_key=avalai_api)

    prompt_template = """
    تو یک دستیار هستی که پاسخ سوالات مطرح شده را بر اساس مقاله و متن داده شده می دهی. 
    سوال: {question}
    متن و مقاله مرتبط:
    {context}

    بر اساس متن و مقاله مرتبط یک پاسخ خلاصه، دقیق و مفید به سوال ارائه کن.
    """

    PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # Combine the retriever with LLM in a RetrievalQA pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=openai_llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | openai_llm
    )

    return qa_chain, retriever


# Streamlit app title
st.title("از من بپرس: مجله دقیقه")

# Initialize the cached QA chain and retriever
qa_chain, retriever = initialize_qa_chain()

# Session state to manage inputs and outputs
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'top_docs' not in st.session_state:
    st.session_state.top_docs = ""
if 'reference_url' not in st.session_state:
    st.session_state.reference_url = ""

# Input box for the question
question = st.text_input(
    "درباره 100 مقاله اخیر مجله دقیقه سوال بپرس:",
    value=st.session_state.question,
    placeholder="میزان کشته های تصادفات جاده ای ایران در سال 1402 چقدر بوده؟"
)

# Buttons for execution and clearing
col1, col2 = st.columns(2)

with col1:
    if st.button("جواب بده"):
        qq = question.strip()
        if qq:
            # Retrieve documents with similarity scores
            #retrieved_docs = retriever.get_relevant_documents(qq)
            retrieved_docs = retriever.vectorstore.similarity_search_with_score(qq)
        
            # Format the retrieved documents and similarity scores
            top_docs = "\n".join(
                [f"Score: {doc[1]}\nContent: {doc[0].page_content}\nLink: {doc[0].metadata['url']}\n" for doc in retrieved_docs]
            )
            st.session_state.top_docs = top_docs

            # Run the QA chain to get the answer
            answer = qa_chain.invoke(qq)
            st.session_state.answer = answer.content
            st.session_state.question = question
        else:
            st.warning("متنی بعنوان سوال بنویسید.")

with col2:
    if st.button("پاک کن"):
        st.session_state.question = ""
        st.session_state.answer = ""
        st.session_state.top_docs = ""
        st.session_state.reference_url = ""

# Display the retrieved documents with similarity scores
if st.session_state.top_docs:
    st.text_area("مقالات مرتبط و امتیاز شباهت:", st.session_state.top_docs, height=300)

# Display the answer
if st.session_state.answer:
    st.text_area("پاسخ:", st.session_state.answer, height=100)
    if st.session_state.reference_url:
        st.markdown(f"مرجع: [کلیک کنید]({st.session_state.reference_url})")
