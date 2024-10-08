#Hello! It seems like you want to import the Streamlit library in Python. Streamlit is a powerful open-source framework used for building web applications with interactive data visualizations and machine learning models. To import Streamlit, you'll need to ensure that you have it installed in your Python environment.
#Once you have Streamlit installed, you can import it into your Python script using the import statement,

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch.nn.functional as F
import joblib

import streamlit as st


import pandas as pd
from io import StringIO

import joblib

import torch
import transformers
import langchain

# from langchain_openai import OpenAI

#When deployed on huggingface spaces, this values has to be passed using Variables & Secrets setting, as shown in the video :)
#import os
#os.environ["OPENAI_API_KEY"] = "sk-PLfFwPq6y24234234234FJ1Uc234234L8hVowXdt"

from langchain.llms import HuggingFaceEndpoint

from langchain import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# llm = HuggingFaceEndpoint(repo_id="tiiuae/falcon-7b")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3") 

# llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct") # Last used model 
 
# llm = HuggingFaceEndpoint(repo_id="google-bert/bert-base-uncased") 

# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# llm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")



# llm = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

#Function to return the response
# def load_answer(question):
#     # llm = OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0)
#     llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

#     answer=llm.invoke(question)
#     return answer


#App UI starts here
st.set_page_config(page_title="Seismic Metadata", page_icon=":robot:")
st.header("Seismic Metadata Management (SMM) app")

st.subheader("Predict Survey Name using a non-tuned LLM model:")

#Gets the user input
def get_text():
    # input_text = st.text_input("You: ", key="input")
    input_text = st.text_area("Enter the seismic file header (EBCDIC) here: ", key='input', height=100)
    return input_text


user_input=get_text()

# encoded_input = tokenizer(user_input, return_tensors='pt')
# output = llm(**encoded_input)

template = """
{our_text}

Can you find the survey name in the above text? Just give the answer and not the explaination. Put the answer in quotations.

Give only one word answer.
"""

prompt = PromptTemplate(
    input_variables=["our_text"],
    template=template)

final_prompt = prompt.format(our_text=user_input)

# response = load_answer(user_input)

submit = st.button('Find the Survey Name')  

# If generate button is clicked
if submit:

    st.subheader("Survey Name:")

    # st.write(final_prompt)
    st.write(llm.invoke(final_prompt))

    # response = llm(final_prompt)

    # st.write(response)



                                                                                    # LLM Classifier model:

st.subheader("Predict Survey Name using a fine-tuned LLM model:")


#Gets the user input
def get_text_2():
    # input_text = st.text_input("You: ", key="input")
    input_text_2 = st.text_area("Enter the seismic file header (EBCDIC) here: ", key='input_2', height=100)
    return input_text_2


user_input_2=get_text_2()


def predict_survey_name(metadata):
    # Tokenize the input metadata
    encoding = tokenizer.encode_plus(
        metadata,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move input tensors to the same device as the model
    input_ids = encoding['input_ids'].to(device) 
    attention_mask = encoding['attention_mask'].to(device) 

    # Set the model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        # Perform the forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Extract logits and move them to CPU for processing
        logits = outputs.logits.detach().cpu()  # Ensure logits are detached and on CPU
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get the predicted class index and its probability
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_index].item() * 100  # Convert to percentage
    

        # # Get the predicted label by finding the index of the max logit
        # _, prediction = torch.max(outputs.logits, dim=1)
        
        
    # # Convert the predicted index to the actual label using the LabelEncoder
    # predicted_label = label_encoder.inverse_transform([prediction.item()])[0]
    
    # Convert the predicted index to the actual label using the LabelEncoder
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label, confidence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_directory = "D:\Seismic Dropsite Metadata Management\savel_model_bert_run_02\SMM\saved_model"


# Load the saved model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained(save_directory).to(device)
tokenizer = BertTokenizer.from_pretrained(save_directory)
label_encoder = joblib.load(os.path.join(save_directory, 'label_encoder.joblib'))

print("Model, tokenizer, and label encoder loaded successfully.")


button = st.button("Predict", key = 12345)

if button:


    st.subheader("Predicted survey name")

    prediction, CL = predict_survey_name(user_input_2)

    # st.write(final_prompt)
    st.write("The predicted Survey Name is : ", prediction)

    st.write("The confidence level is: ", str(round(CL,2))+"%"
                                              )







                                                                                    # CatBoost Classifier model:

st.subheader("Predict Survey Name using a CatBoost model:")


#Gets the user input
def get_text_3():
    # input_text = st.text_input("You: ", key="input")
    input_text_3 = st.text_area("Enter the seismic file header (EBCDIC) here: ", key='input_3', height=100)
    return input_text_3


user_input_3=get_text_3()


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib  # For saving and loading model and preprocessing objects


model_dir = "D:\Seismic Dropsite Metadata Management\Code\model_catboost"

# Step 11: Load the model, vectorizer, and label encoder
def load_model(model_dir):
    """
    Load the saved CatBoost model, TF-IDF vectorizer, and LabelEncoder.
    
    :param model_dir: Directory where the model and files are saved
    :return: Loaded model, vectorizer, and label encoder
    """
    # Load the CatBoost model
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(os.path.join(model_dir, "catboost_model.cbm"))
    
    # Load the vectorizer
    loaded_vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    
    # Load the label encoder
    loaded_label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    
    return loaded_model, loaded_vectorizer, loaded_label_encoder

# Load the model, vectorizer, and label encoder
loaded_model, loaded_vectorizer, loaded_label_encoder = load_model(model_dir)
print("\nModel, vectorizer, and label encoder loaded successfully.")

# Step 12: Inference - Predicting the Label
def predict_label(text, model, vectorizer, label_encoder):
    """
    Predict the label for a given text using the trained model.
    
    :param text: Input text to classify
    :param model: Trained CatBoost model
    :param vectorizer: TF-IDF vectorizer used during training
    :param label_encoder: LabelEncoder used during training
    :return: Predicted label
    """
    # Transform the input text using the trained vectorizer
    text_tfidf = vectorizer.transform([text])
    
    # Predict the probabilities for each class
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Find the index of the maximum probability
    predicted_label_index = np.argmax(probabilities)
    
    # Decode the predicted label index to the original label
    predicted_label = label_encoder.inverse_transform([int(predicted_label_index)])[0]
    
    # Get the confidence level for the predicted label
    confidence_level = round(probabilities[predicted_label_index] * 100,2)  # in percentage
    
    return predicted_label, confidence_level


button = st.button("Predict", key = 123456)

if button:


    st.subheader("Predicted survey name")

    prediction, CL = predict_label(user_input_3, loaded_model, loaded_vectorizer, loaded_label_encoder)

    # st.write(final_prompt)
    st.write("The predicted Survey Name is : ", prediction)

    st.write("The confidence level is: ", str(round(CL,2))+"%"
                                              )




