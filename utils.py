import os
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM

import gradio as gr
import datetime

def set_openai_api_key(api_key):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    MAX_TOKENS = 512
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key
        print("\n\n ++++++++++++++ Setting OpenAI API key ++++++++++++++ \n\n")
        print(str(datetime.datetime.now()) + ": Before OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        prompt_template = """Please answer the user's question about document.
        Question: {question}
        Answer:"""


        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

        llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKENS, model_name="gpt-3.5-turbo")


        # Pertains to question answering functionality
        base_embeddings = OpenAIEmbeddings()

        print(str(datetime.datetime.now()) + ": After load_chain, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        os.environ["OPENAI_API_KEY"] = ""
        return llm, base_embeddings
    return None, None
    

    
    
def add_text(history, text,llm):
    if not llm:
        raise gr.Error("OpenAI key not found!")
    history = history + [(text, None)]
    return history, ""


def bot(history,llm,docsearch):
    question = history[-1][0]
    response = qna_with_doc(question,llm,docsearch)
    history[-1][1] = response
    return history

def qna_with_doc(question,llm, docsearch):
    try:
        if not docsearch:
            raise gr.Error("Knowledge base not found!")
        print(docsearch)
            
        retriever = docsearch.as_retriever()
        retriever.search_kwargs = {"k":5}
        qa = RetrievalQA.from_chain_type(llm=llm, 
                                         chain_type="stuff",
                                         retriever=retriever,
                                        )
        out_text= qa.run(question)
    
    except Exception as e:
        raise gr.Error(f"Error: {e}")
        out_text = f"Error: {e}"
    return out_text



def prepare_uploaded_file(files,llm,base_embeddings,docsearch, preload=False):
    if not llm:
        raise gr.Error("OpenAI key not found!")
    if preload:
        full_text = open(files,'r').read()
    else:
        full_text  = open(files[0].name,'r').read()
        full_text = full_text.replace("ABC app","SBNRI app")
        
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=30)
    texts = text_splitter.split_text(full_text)
    print(f"Splitted to {len(texts)} token")
    docsearch = Chroma.from_texts(texts,
                                  base_embeddings)
    
    return docsearch, "File Loaded Successfully!"


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths
