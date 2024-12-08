
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch.nn.functional as F
import streamlit as st
import pandas as pd
from io import StringIO
import joblib
import transformers
import langchain
from langchain.llms import HuggingFaceEndpoint
from langchain import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import requests 

#App UI starts here
st.set_page_config(page_title="Seismic Metadata QC", page_icon=":robot:")
st.header("Seismic Metadata QC app")

st.write("Discover a smarter way to enhance seismic data integrity with our innovative app, tailored specifically for seismic surveys in the Norway (North Sea) region. This cutting-edge solution leverages advanced machine learning models to refine and correct critical metadata, starting with "Survey Name" information.

Simply input an EBCDIC header, and our app will evaluate and improve the quality of your "Survey Name" metadata, providing an accuracy score for added confidence. Designed to optimize data management for seismic professionals, this app ensures your metadata meets the highest standards, paving the way for more reliable analysis and decision-making.

While currently focused on "Survey Name" metadata, the app is built to scale, offering the potential to address a wide range of metadata attributes in the future. Whether you're streamlining existing data or preparing for expansive metadata corrections, our app is the perfect companion for quality-conscious geophysical workflows.

Elevate your seismic metadata today—because precision starts here.")

st.subheader("Using prompt engineered LLM model:")


# Function to validate Huggingface API key
def validate_huggingface_api_key(api_key):
    """
    Validates the provided Hugging Face API key by making a test request.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
    
    # Return True if the response is 200, otherwise False
    if response.status_code == 200:
        return True
    else:
        return False

hug_api = st.sidebar.text_input('Huggingface API Key:', type='password') 


#Gets the user input
def get_text():
    # input_text = st.text_input("You: ", key="input")
    input_text = st.text_area("Enter your seismic (EBCDIC) header here: ", key='input', height=100)
    return input_text


user_input=get_text()


template = """
{our_text}

Can you find the survey name in the above text? Just give the answer and not the explanation. Put the answer in quotations.

Give only one word answer.
"""

prompt = PromptTemplate(
    input_variables=["our_text"],
    template=template)

final_prompt = prompt.format(our_text=user_input)

# response = load_answer(user_input)

submit = st.button('Find the Survey Name')  

if submit:
    if not hug_api:
        st.error("Please provide a valid Huggingface API Key before adding data.")
    else:
        is_valid = validate_huggingface_api_key(hug_api)
        if not is_valid:
            st.error("Invalid Huggingface API Key. Please provide a correct key.")
        else:
            llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", token = hug_api) 
  
            st.subheader("Survey Name:")
        
            # st.write(final_prompt)
            st.write(llm.invoke(final_prompt))
        
            # response = llm(final_prompt)
        
            # st.write(response)



                                                                                    # LLM Classifier model:

st.subheader("Using a fine-tuned LLM model:")


#Gets the user input
def get_text_2():
    # input_text = st.text_input("You: ", key="input")
    input_text_2 = st.text_area("Enter your seismic (EBCDIC) header here: ", key='input_2', height=100)
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

save_directory = "saved_model"

# Load the saved model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained(save_directory).to(device)
tokenizer = BertTokenizer.from_pretrained(save_directory)
label_encoder = joblib.load(os.path.join(save_directory, 'label_encoder.joblib'))

print("Model, tokenizer, and label encoder loaded successfully.")


button = st.button("Predict", key = 12345)

if button:
    st.subheader("Predicted survey name:")

    prediction, CL = predict_survey_name(user_input_2)

    # st.write(final_prompt)
    st.write("The predicted Survey Name is : ", prediction)

    st.write("The confidence level is: ", str(round(CL,2))+"%")
                                              


                                                                                    # CatBoost Classifier model:

st.subheader("Using a classic ML model:")


#Gets the user input
def get_text_3():
    # input_text = st.text_input("You: ", key="input")
    input_text_3 = st.text_area("Enter your seismic (EBCDIC) header here: ", key='input_3', height=100)
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


model_dir = "model_catboost"

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
    st.subheader("Predicted survey name:")

    prediction, CL = predict_label(user_input_3, loaded_model, loaded_vectorizer, loaded_label_encoder)

    # st.write(final_prompt)
    st.write("The predicted Survey Name is : ", prediction)

    st.write("The confidence level is: ", str(round(CL,2))+"%"
                                              )




