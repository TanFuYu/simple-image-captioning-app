import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch(server_name="0.0.0.0", server_port= 7860) # server_name="0.0.0.0" → allows other devices on the same network to access your app even if the app is local. Other devices on the same network (same Wi-Fi/LAN) can access it using your computer’s IP address