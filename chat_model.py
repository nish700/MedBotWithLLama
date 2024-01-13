from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

import chainlit as cl

import warnings
warnings.filterwarnings("ignore")

# path where the vector databsase is stored
DB_FAISS_PATH = 'vectorstore/db_faiss/all-mpnet-base-v2'
# prompt template
custom_prompt_template = """Use the following piece of information to answer the user's question. 
If you don't know the answer, just say Unable to find the response and don't try to make up an answer. 

Context : {context}
Question : {question}

Only return the helpful answer below and nothing else:
Helpful Answer: 

"""

# set the custom prompt for the Llama 2 model
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template = custom_prompt_template,
                            input_variables = ['context', 'questions'])
    return prompt


# Retrieval QA chain
def retrieval_QA_chain(llm, prompt, db):

    qa_chain = RetrievalQA.from_chain_type(
        llm= llm,
        chain_type = 'stuff',
        retriever = db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': prompt},
        verbose= False
    )

    return qa_chain

# loading the model
def load_llm():
    # load the locally downloaded model
    config = {
        'max_new_tokens' : 512,
        'context_length':1248,
        'temperature' : 0.2,
        'repetition_penalty':2
    }
    llm = CTransformers(
        model = "pretrained_model/8-bit/llama-2-7b-chat.Q5_K_M.gguf",
        model_type = "llama",
        config = config
    )
    return llm


# QA model function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs = {'device':'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_QA_chain(llm, qa_prompt, db)

    return qa

# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# chainlit code to start the chatbot
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot....")
    await msg.send()
    msg.content = "Hi, Welcome to the medical bot. What is your health query?"
    await msg.update()

    cl.user_session.set("chain", chain)

# chainlit code to retrieve the response from model
@cl.on_message
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=['FINAL', 'ANSWER']
    )

    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res['result']
    sources = res['source_documents']

    if sources:
        answer += f"\n Sources:" + str(sources)
    else:
        answer += "\n No Source Found"

    await cl.Message(content=answer).send()