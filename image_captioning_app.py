import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def caption_image(input_image: np.ndarray): # : np.ndarray = type hint
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image
    text = "the image of " # The BLIP model treats text as: “Here’s some context / hint — now generate a sentence describing the image.”
    inputs = processor(images=raw_image, text=text, return_tensors="pt")

    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text and store it into `caption`
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(), # Gradio will throw the input to the first argument of the function i.e. fn
    outputs="text", # Similarly the output return by the fn will be thrown to output as Gradio takes this returned value and displays it in the output component defined. Since outputs="text", Gradio will render the string as plain text below the input box.
    # Gradio expects the return type to match the output component if not it will raise an error e.g. TypeError: Cannot convert <class 'str'> to image
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch() # Gradio will automatically pick a free port on your machine if its not defined
