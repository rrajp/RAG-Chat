def set_openai_api_key(api_key):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
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
    
    
import gradio as gr

gr.close_all()
demo = gr.Blocks()

import requests
import re

import os


from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


import openai
import datetime
 
MAX_TOKENS = 512


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
        retriever = docsearch.as_retriever()
        retriever.search_kwargs = {"k":5}
        qa = RetrievalQA.from_chain_type(llm=llm, 
                                         chain_type="stuff",
                                         retriever=retriever,
                                        )
        out_text= qa.run(question)
    
    except Exception as e:
        gr.Error(f"Error in QNA : {e}")
        out_text = f"Error in QNA : {e}"
    return out_text



def prepare_uploaded_file(files,llm,base_embeddings,preload=False):
    global docsearch
    if not llm:
        raise gr.Error("OpenAI key not found!")
    if preload:
        full_text = open(files,'r').read()
    else:
        full_text  = open(files[0].name,'r').read()
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=30)
    texts = text_splitter.split_text(full_text)
    print(f"Splitted to {len(texts)} token")
    docsearch = Chroma.from_texts(texts,
                                  base_embeddings)
    
    return docsearch, "File Loaded Successfully!"


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

demo = gr.Blocks()


with demo:
    base_embeddings = gr.State()
    llm = gr.State()
    docsearch = gr.State()
    
    gr.Markdown(
        "<h1><center> Not a bot - bot! </center></h1>")
    with gr.Row():
        with gr.Tabs(scale = 0.1):
            with gr.TabItem("Question Answer"):
                chatbot = gr.Chatbot([], elem_id = "chatbot").style(height = 350)

                with gr.Row():
                    with gr.Column(scale = 0.1, min_width = 0):
                        clear = gr.Button("Clear")
                    with gr.Column(scale = 0.9):
                        chat_txt = gr.Textbox(
                            show_label = False,
                            placeholder = "Enter text and press enter, or upload an image",
                        ).style(container = False)
                        clear.click(lambda: None, None, chatbot, queue = False)

#                 chat_txt.submit(add_text, [chatbot, chat_txt], [chatbot, chat_txt]).then(
#                     bot, chatbot, chatbot
#                 )
        with gr.Tabs(scale = 0.9):
            with gr.TabItem("Settings"):
                openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...) and hit Enter",
                                                show_label=False, lines=1, type='password')

#                 gr.HTML("Preloaded File: KnowledgeDocument(pan_card_ser...")
                file = gr.File(file_count = 1,
                               label = "Update Knowledge File")
                html_doc = gr.HTML("")
                with gr.Column():
                    upload_button = gr.Button("Upload File")
                    process_button = gr.Button("Process File for chat")

    chat_txt.submit(add_text, [chatbot, chat_txt,llm], [chatbot, chat_txt]).then(
                    bot, [chatbot,llm, docsearch], chatbot
                )
    clear.click(lambda: None, None, chatbot, queue = False)
    upload_button.click(prepare_uploaded_file, inputs=[file,llm, base_embeddings], outputs = [docsearch,html_doc])
    
    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox],
                                  outputs=[llm, base_embeddings])

#     prepare_uploaded_file("/home/ravirajprajapat/Downloads/KnowledgeDocument(pan_card_services).txt",
#                        preload=True)
demo.launch(enable_queue = True, debug = True)#, server_name = "0.0.0.0", server_port = 8861)

