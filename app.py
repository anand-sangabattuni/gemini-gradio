import base64
import os
import google.generativeai as genai
import PIL.Image
import gradio as gr

def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

# Set Google API key
os.environ['GOOGLE_API_KEY'] = "YOUR_API_KEY"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


# Create the Model
txt_model = genai.GenerativeModel('gemini-pro')
vis_model = genai.GenerativeModel('gemini-pro-vision')

def llm_response(history,text,img):
    if not img:
        response = txt_model.generate_content(text)
        history += [(None,response.text)]
        return history
    else:
        img = PIL.Image.open(img)
        response = vis_model.generate_content([text,img])
        history += [(None,response.text)]
        return history
    
# Function that takes User Inputs and displays it on ChatUI
def query_message(history,txt,img):
    if not img:
        history += [(txt,None)]
        return history
    base64 = image_to_base64(img)
    data_url = f"data:image/jpeg;base64,{base64}"
    history += [(f"{txt} ![]({data_url})", None)]

# UI Code
with gr.Blocks() as app:
    with gr.Row():
        image_box = gr.Image(type="filepath",scale =3, height= 300)

        chatbot = gr.Chatbot(
            scale = 3,
            height= 300
        )
    text_box = gr.Textbox(
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )


    btn = gr.Button("Submit")
    clicked = btn.click(query_message,
                        [chatbot,text_box,image_box],
                        chatbot
                        ).then(llm_response,
                                [chatbot,text_box,image_box],
                                chatbot
                                )
app.queue()
app.launch(debug=True)
