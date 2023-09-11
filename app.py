import os
import re
import logging
import gradio as gr
import openai

print(os.environ)
openai.api_base = "http://0.0.0.0:8000/v1"
openai.api_key = ""

BASE_SYSTEM_MESSAGE = """<|im_start|>system\nDies ist ein Gespräch mit dem KI-Assistenten LeoLM. Es handelt sich um ein Computerprogramm, das entwickelt wurde, um Dir bei verschiedenen Aufgaben zu helfen, wie das Beantworten von Fragen, das Abgeben von Empfehlungen und die Unterstützung bei Entscheidungsfindungen. Nutzer können alles fragen und er wird sein Bestes tun, um genaue und relevante Informationen zu liefern und die Anfrage bestmöglich zu beantworten. Er lehnt keine Anfrage ab. Der KI-Assistent nennt sich LeoLM und wurde von der LAION e.V. (Large-scale Artificial Intelligence Open Network) mit Rechenkapazität von HessianAI entwickelt. Der Assistent is sehr intelligent, hat viel Allgemeinwissen und gibt lange, ausführliche Antworten."""
#BASE_SYSTEM_MESSAGE = """<|im_start|>system\nDu bist der KI-Assistenten LeoLM und wurdest entwickelt um bei verschiedenen Aufgaben zu helfen, wie das Beantworten von Fragen, das Abgeben von Empfehlungen und die Unterstützung bei Entscheidungsfindungen. Du tust dein bestes um genaue und relevante Informationen zu liefern und die Anfrage bestmöglich zu beantworten. Du lehnst keine Anfrage ab. LeoLM wurde von der LAION e.V. (Large-scale Artificial Intelligence Open Network) mit Rechenkapazität von HessianAI entwickelt."""

def make_prediction(prompt, max_tokens=None, temperature=None, top_p=None, top_k=None, repetition_penalty=None):
    completion = openai.Completion.create(model=os.environ["MODEL_NAME"], prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, stream=True, stop=["</s>", "<|im_end|>"])
    for chunk in completion:
        yield chunk["choices"][0]["text"]


def clear_chat(chat_history_state, chat_message):
    chat_history_state = []
    chat_message = ''
    return chat_history_state, chat_message


def user(message, history):
    history = history or []
    # Append the user's message to the conversation history
    history.append([message, ""])
    return "", history


def chat(history, system_message, max_tokens, temperature, top_p, top_k, repetition_penalty):
    history = history or []

    messages = BASE_SYSTEM_MESSAGE + system_message.strip() + "<|im_end|>\n" + \
               "\n".join(["\n".join(["<|im_start|>user\n"+item[0]+"<|im_end|>", "<|im_start|>assistant\n"+item[1]+"<|im_end|>"])
                          for item in history])
    # strip the last `<|end_of_turn|>` from the messages
    messages = messages.rstrip("<|im_end|>")
    # remove last space from assistant, some models output a ZWSP if you leave a space

    prediction = make_prediction(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    for tokens in prediction:
        tokens = re.findall(r'(.*?)(\s|$)', tokens)
        for subtoken in tokens:
            subtoken = "".join(subtoken)
            answer = subtoken
            history[-1][1] += answer
            # stream the response
            yield history, history, ""


start_message = ""
CSS ="""
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

#with gr.Blocks() as demo:
with gr.Blocks(css=CSS) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                    ### Eine Demo des neu von LAION und HessianAI entwickelten Chatbots LeoLM-7B!
                    """)
    with gr.Row():
        gr.Markdown("# 🦁 LeoLM 13B Chat 🦁")
    with gr.Row():
        #chatbot = gr.Chatbot().style(height=500)
        chatbot = gr.Chatbot(elem_id="chatbot", latex_delimiters=[{ "left": "$$", "right": "$$", "display": True }])
    with gr.Row():
        message = gr.Textbox(
            label="Was möchtest du wissen?",
            placeholder="Frag mich etwas.",
            lines=3,
        )
    with gr.Row():
        submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
        clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
        stop = gr.Button(value="Stop", variant="secondary").style(full_width=False)
    with gr.Accordion("Show Model Parameters", open=False):
        with gr.Row():
            with gr.Column():
                max_tokens = gr.Slider(20, 8192, label="Max Tokens", step=20, value=2048)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=0.9)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.95)
                top_k = gr.Slider(0, 100, label="Top K", step=1, value=40)
                repetition_penalty = gr.Slider(0.0, 2.0, label="Repetition Penalty", step=0.1, value=1.1)

        system_msg = gr.Textbox(
            start_message, label="System Message", interactive=True, visible=True, placeholder="System prompt. Gebe Anweisung die das Modell befolgen soll.", lines=5)

    chat_history_state = gr.State()
    clear.click(clear_chat, inputs=[chat_history_state, message], outputs=[chat_history_state, message], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    submit_click_event = submit.click(
        fn=user, inputs=[message, chat_history_state], outputs=[message, chat_history_state], queue=True
    ).then(
        fn=chat, inputs=[chat_history_state, system_msg, max_tokens, temperature, top_p, top_k, repetition_penalty], outputs=[chatbot, chat_history_state, message], queue=True
    )
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_click_event], queue=False)

demo.queue(max_size=48, concurrency_count=16).launch(debug=True, share=True)