import streamlit as st
from google import genai
import chromadb
from pypdf import PdfReader

# Page configuration
st.set_page_config(page_title="PDF RAG Q&A with Gemini", page_icon="📄", layout="wide")

st.title("📄 Private Document Q&A (RAG)")
st.write("Upload a PDF document and ask questions. Gemini will answer strictly based on the text inside your document.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.divider()
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize ChromaDB client (Ephemeral/In-memory for session)
@st.cache_resource
def get_chroma_client():
    return chromadb.EphemeralClient()

chroma_client = get_chroma_client()

# Main application logic
if uploaded_file is not None:
    # Read PDF text using pypdf
    try:
        reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n"
        
        if not pdf_text.strip():
            st.error("Could not extract text from this PDF. It might be scanned images.")
        else:
            # Simple text chunking (split every ~1000 characters with overlap)
            chunk_size = 1000
            chunk_overlap = 200
            chunks = []
            for i in range(0, len(pdf_text), chunk_size - chunk_overlap):
                chunk = pdf_text[i:i + chunk_size]
                chunks.append(chunk)

            # Store chunks into a temporary ChromaDB collection
            collection_name = "pdf_rag_collection"
            # Re-create collection safely
            try:
                chroma_client.delete_collection(name=collection_name)
            except Exception:
                pass
            
            collection = chroma_client.create_collection(name=collection_name)
            
            # Add chunks to Chroma (Chroma handles default embeddings automatically)
            chunk_ids = [f"id_{i}" for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                ids=chunk_ids
            )
            
            st.sidebar.success(f"Successfully processed PDF into {len(chunks)} chunks!")

            # User Question Input
            user_query = st.text_input("Ask a question about your PDF document:")

            if user_query:
                if not api_key:
                    st.error("Please enter your Gemini API Key in the sidebar.")
                else:
                    with st.spinner("Searching document & generating answer..."):
                        # 1. Retrieve top 3 relevant chunks from ChromaDB
                        results = collection.query(
                            query_texts=[user_query],
                            n_results=min(3, len(chunks))
                        )
                        retrieved_chunks = results["documents"][0]
                        context = "\n\n".join(retrieved_chunks)

                        # 2. Construct RAG Prompt for Gemini
                        prompt = f"""You are a precise assistant. Answer the user's question using ONLY the context provided below. If the answer cannot be found in the context, say "I cannot find the answer in the uploaded document."

Context from Document:
{context}

User Question: {user_query}
Answer:"""

                        # 3. Call Gemini API
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt,
                        )

                        st.subheader("Answer:")
                        st.write(response.text)

                        with st.expander("View Retrieved Document Context"):
                            st.write(context)

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
else:
    st.info("👈 Please upload a PDF file in the sidebar to get started.")