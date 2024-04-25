# !pip install transformers
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# specify model to be used
hf_model = "Salesforce/blip-image-captioning-large"
# use GPU if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocessor will prepare images for the model
processor = BlipProcessor.from_pretrained(hf_model)
# then we initialize the model itself
model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)
# import requests
from PIL import Image




def picturedescripe(imgurl):
    raw_image =  Image.open(imgurl).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    print(processor.decode(out[0], skip_special_tokens=True))
    return processor.decode(out[0], skip_special_tokens=True)
# des=picturedescripe("/root/show/b.png")
# from JudgeLLM import Translatechain
# final=Translatechain.invoke({"input": des})
# print(final)
