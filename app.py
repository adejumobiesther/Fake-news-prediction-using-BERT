import streamlit as st
import config
import torch
import time
from model import BERTBaseUncased
import functools
import torch.nn as nn


# set page config
st.set_page_config(
	page_title="Analyze US Political News that are True or Fake",
	page_icon="🏨"
)

DEVICE = config.DEVICE
PREDICTION_DICT = dict()

# load model

with st.spinner("Loading our awesome AI 🤩. Please wait ..."):  
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()

@st.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    news = str(sentence)
    news = " ".join(news.split())

    inputs = tokenizer.encode_plus(
        news, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

st.title("🏨 US politics Fake News Prediction")
st.write("Do you want to verify if a news is real or fake?")
st.write("Checking for fake news is not an easy task, let our AI do it for you! 😆")
st.write("It's easy and fast. Put the news down below and we will take care the rest 😉")

# user input
news = st.text_area(
	label="News:",
	help="Input the text here, then click anywhere outside the box."
)

def predict(news):
    negative_prediction = sentence_prediction(news)
    positive_prediction = 1 - negative_prediction
    response = {}
    response["response"] = {
        "Real": positive_prediction,
        "Fake": negative_prediction,
        "news": str(news)
    }
    return response


if news != "":
    prediction = predict(news)

    # extract predicted class
    predicted_class = "Real" if prediction["response"]["Real"] > prediction["response"]["Fake"] else "Fake"

    # display prediction
    st.subheader("AI thinks that ...")

    st.write(f"Our model predicted that the news is {predicted_class.lower()}")