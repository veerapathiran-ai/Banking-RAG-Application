import os
import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load AWS Credentials and Region from .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize AWS S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

app = FastAPI(title="Amazon Bedrock Nova Pro RAG API", version="3.2")

UPLOAD_DIR = "data/uploads"
FAISS_DIR = "data/faiss_index"

for path in [UPLOAD_DIR, FAISS_DIR]:
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)

# Local HuggingFace Embeddings (384 Dimensions)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Amazon Bedrock Runtime & Nova Pro LLM via Converse API
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

llm = ChatBedrockConverse(
    client=bedrock_runtime,
    model_id="amazon.nova-pro-v1:0",
    temperature=0.0
)

def get_vectorstore():
    index_file = os.path.join(FAISS_DIR, "index.faiss")
    if os.path.exists(index_file):
        return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    return None

@app.post("/upload-to-s3")
async def upload_to_s3(file: UploadFile = File(...)):
    try:
        s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, file.filename)
        return {"message": f"Successfully uploaded '{file.filename}' to AWS S3 bucket!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 Upload failed: {str(e)}")

@app.post("/sync-s3")
async def sync_s3_document():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if "Contents" not in response:
            raise HTTPException(status_code=400, detail="No files found in S3 bucket.")
        
        all_docs = []
        downloaded_files = []
        
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            if s3_key.lower().endswith(".pdf"):
                local_file_path = os.path.join(UPLOAD_DIR, os.path.basename(s3_key))
                s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path)
                
                loader = PyPDFLoader(local_file_path)
                docs = loader.load()
                all_docs.extend(docs)
                downloaded_files.append(s3_key)
        
        if not all_docs:
            raise HTTPException(status_code=400, detail="No PDF files found in S3 bucket.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_DIR)
        
        return {"message": f"Successfully synced files ({', '.join(downloaded_files)}) from S3 and indexed locally!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 Sync failed: {str(e)}")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            raise HTTPException(status_code=400, detail="No documents indexed yet. Please sync from S3 first.")
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Retrieve relevant context documents directly
        docs = retriever.invoke(request.question)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        system_prompt = (
            "You are an expert enterprise AI assistant. Answer the user's question "
            "strictly using the provided context below. If you do not know the answer, "
            "say that you don't know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {input}"),
        ])
        
        # Format prompt messages with context and invoke LLM directly
        formatted_messages = prompt.format_messages(context=context_text, input=request.question)
        response = llm.invoke(formatted_messages)
        
        return {"answer": response.content}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")