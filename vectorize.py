from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


def upload_documents(DATA_PATH):
    #upload multiple pdf files
    loader = PyPDFDirectoryLoader(DATA_PATH,glob='*.pdf')

    documents = loader.load()
    return documents

# create vector database
def create_vectorDB(DATA_PATH,chunk_size,chunk_overlap):
    # split the text in the document
    documents = upload_documents(DATA_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap= chunk_overlap,
                                                   separators=["\n\n","\n","\t"," ",""],
                                                   length_function=len,
                                                   add_start_index= True
                                                   )
    #split the data into chunks
    texts_chunks = text_splitter.split_documents(documents)

    return texts_chunks

# get vector store and perform similarity search
def get_vectorstore(chunk_size,chunk_overlap):
    texts = create_vectorDB(DATA_PATH, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs = {'device': 'cuda'})

    # Facebook AI Similarity Search (FAISS) is a library that enables efficient similarity search of vectors
    db = FAISS.from_documents(texts, embeddings)

    # save the embeddings
    save_path = os.path.join(DB_FAISS_PATH ,'all-mpnet-base-v2' )
    db.save_local(save_path)
    return db


if __name__=="__main__":
    get_vectorstore(chunk_size=1000, chunk_overlap=200)

