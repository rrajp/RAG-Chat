import gradio as gr

gr.close_all()
demo = gr.Blocks()

import requests
import re
import os

import openai
import whisper


from utils import set_openai_api_key, add_text, bot, qna_with_doc, prepare_uploaded_file, upload_file
 
MAX_TOKENS = 512
WHISPER_MODEL = whisper.load_model("tiny")

def audio_transcribe(aud_inp, whisper_lang):
    if aud_inp is None:
        return ""
    aud = whisper.load_audio(aud_inp)
    aud = whisper.pad_or_trim(aud)
    mel = whisper.log_mel_spectrogram(aud).to(WHISPER_MODEL.device)
    _, probs = WHISPER_MODEL.detect_language(mel)
    options = whisper.DecodingOptions(fp16 = False)
    
    result = whisper.decode(WHISPER_MODEL, mel, options)
    print("result.text", result.text)
    result_text = ""
    if result and result.text:
        result_text = result.text
    return result_text


demo = gr.Blocks()


with demo:
    base_embeddings = gr.State()
    llm = gr.State()
    docsearch = gr.State(None)
    whisper_lang_state = gr.State('Detect language')
    
    gr.Markdown(
        "<h1><center> RAG Bot! </center></h1>")
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
                audio_comp = gr.Microphone(source="microphone", type="filepath", label="Please Speak !", interactive=True, streaming=False)
                audio_comp.change(audio_transcribe, inputs=[audio_comp, whisper_lang_state], outputs=[chat_txt])
                

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
    upload_button.click(prepare_uploaded_file, inputs=[file,llm, base_embeddings, docsearch], outputs = [docsearch,html_doc])
    
    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox],
                                  outputs=[llm, base_embeddings])

demo.launch(enable_queue = True, debug = True)#, server_name = "0.0.0.0", server_port = 8861)

