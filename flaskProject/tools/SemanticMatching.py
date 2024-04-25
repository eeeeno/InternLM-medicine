from transformers.utils import logging
import torch
logging.set_verbosity_error()
from transformers import BlipForImageTextRetrieval

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(device)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-itm-base-coco")
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
from PIL import Image
# import requests
# imagePath="/root/show/a.jpg"
# raw_image =  Image.open(imagePath).convert('RGB')
# text = "The majority of the photo consists of human skin"
# text="This is a drug leaflet"
# inputs = processor(images=raw_image,
#                    text=text,
#                    return_tensors="pt").to(device)
# itm_scores = model(**inputs)[0]

# itm_score = torch.nn.functional.softmax(
#     itm_scores,dim=1)
# print(f"""\
# The image and text are matched \
# with a probability of {itm_score[0][1]:.4f}""")
def semanticMatching(imgurl):
    raw_image =  Image.open(imgurl).convert('RGB')
    text1 = "The majority of the photo consists of human skin"
    # text2="There is a bunch of text in the picture, most likely the drug insert"
    # text3="This image has neither a photograph of people's skin, nor what appears to be a drug leaflet"
    inputs1 = processor(images=raw_image,
                   text=text1,
                   return_tensors="pt").to(device)
    itm_scores1 = model(**inputs1)[0]
    itm_score1 = torch.nn.functional.softmax(
    itm_scores1,dim=1)
    # inputs2 = processor(images=raw_image,
    #                text=text2,
    #                return_tensors="pt").to(device)
    # itm_scores2 = model(**inputs2)[0]
    # itm_score2 = torch.nn.functional.softmax(
    # itm_scores2,dim=1)
    # inputs3 = processor(images=raw_image,
    #                text=text3,
    #                return_tensors="pt").to(device)
    # itm_scores3 = model(**inputs3)[0]
    # itm_score3 = torch.nn.functional.softmax(
    # itm_scores3,dim=1)
    print("分数为")
    print(itm_score1[0][1])
    if itm_score1[0][1]>0.3 :
        return 0
    
    # elif itm_score2[0][1]>itm_score1[0][1] and itm_score2[0][1]>itm_score3[0][1]:
    #     return 1
    else:
        return 1
#     print(itm_score1[0][1],itm_score2[0][1],itm_score3[0][1])

# a=semanticMatching("/root/show/b.png")
# print(a)
#     print(f"""\
# The image and text are matched \
# with a probability of {itm_score1[0][1]:.4f} {itm_score2[0][1]:.4f} {itm_score3[0][1]:.4f} """)
# semanticMatching("/root/show/a.jpg")




