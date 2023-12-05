# # Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import streamlit as st
# from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)


# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#     )

#     st.write("# Welcome to Streamlit! 👋")

#     st.sidebar.success("Select a demo above.")

#     st.markdown(
#         """
#         Streamlit is an open-source app framework built specifically for
#         Machine Learning and Data Science projects.
#         **👈 Select a demo from the sidebar** to see some examples
#         of what Streamlit can do!
#         ### Want to learn more?
#         - Check out [streamlit.io](https://streamlit.io)
#         - Jump into our [documentation](https://docs.streamlit.io)
#         - Ask a question in our [community
#           forums](https://discuss.streamlit.io)
#         ### See more complex demos
#         - Use a neural net to [analyze the Udacity Self-driving Car Image
#           Dataset](https://github.com/streamlit/demo-self-driving)
#         - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
#     """
#     )


# if __name__ == "__main__":
#     run()


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
import nltk
import re
import contractions
import pickle

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from os import path
from PIL import Image



def LowerCase(text):
    #text=[y.lower() for y in text]
    text = text.lower()
    return text

def ContractionExpansion(text):
  text = contractions.fix(text)
  return text

def Tokenization(text):
    text = nltk.word_tokenize(text)
    return text

def PunctuationWhitespaceRemoval(text):
    #Punctuation
    text = [re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), '', token) for token in text]

    #Whitespace
    text = [re.sub('\s+', '', token) for token in text]

    #Exclude unwanted empty string token
    text = [token for token in text if token.strip()]
    return text

def StopwordsRemoval(text):
    stop_words = set(stopwords.words('english'))
    text = [token for token in text if token.lower() not in stop_words]
    return text


# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert POS tags from Treebank to WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def Lemmatization(text):
  # Lemmatize the tokenized words in the 'tokenized_review' column with specified POS tags
  text = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for word, pos_tag in nltk.pos_tag(text)]
  return text

# Function to remove usernames from the text
def UsernameRemoval(text):
    # Define a regular expression pattern to match Twitter handles
    twitter_handle_pattern = r'@[A-Za-z0-9_]+'
    # Use re.sub to replace Twitter handles with an empty string
    text = re.sub(twitter_handle_pattern, '', text)
    return text

def DigitRemoval(text):
    # Remove standalone digits
    text = re.sub(r'\b\d+\b', '', text) #remove digit

    # Remove words containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text) #remove worddigit
    return text

# Function to remove URLs from the tweets
def URLRemoval(text):
  url_pattern = r'https?://\S+|www\.\S+'
  text = re.sub(url_pattern, '', text)
  return text

# Function to remove Non-ASCII characters like 'Ã¯Â¿Â'
def Non_ASCIIRemoval(text):
    text = re.sub(r'[^\x00-\x7F\u00EF\u00BF]+', '', text)
    return text

def EmojiRemove(text):
    # Remove emojis using regex
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
def normalized_sentence(sentence):
    sentence= LowerCase(sentence)
    sentence= ContractionExpansion(sentence)
    sentence= EmojiRemove(sentence)
    sentence= UsernameRemoval(sentence)
    sentence= URLRemoval(sentence)
    sentence= Non_ASCIIRemoval(sentence)
    sentence= DigitRemoval(sentence)

    sentence= Tokenization(sentence)
    sentence= StopwordsRemoval(sentence)
    sentence= PunctuationWhitespaceRemoval(sentence)
    sentence= Lemmatization(sentence)

    return sentence

#---------------------------------------------------------------------------------------------------------------------
#Fitted Tokenizer
MAXLEN = 35
tokenizer = Tokenizer(oov_token='UNK')

# Load the tokenizer from file
with open('https://github.com/jxuanT/fyp-deploy1205/blob/main/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

model = load_model('fyp-deploy1205/best_biLSTM_model.h5', compile=False)

label_encoder = joblib.load('label_encoder.pkl')

emoji = {"anger": "😡", "fear":"😨","joy": "😆","love":"🥰","sadness":"😭","surprise":"😮"}
def main():


  menu = ["About","Data Exploration","Tweets Emotion Detection","Findings"]
  choice = st.sidebar.selectbox("Menu",menu)

  if choice == "Tweets Emotion Detection":
    st.subheader("Predict Your Emotion")
    with st.form (key='emotion_clf_form'):
      raw_text = st.text_area("Type Here")
      submit_text = st.form_submit_button(label='Enter')

    if submit_text:
      col1,col2 = st.columns(2)
      normalized_text = normalized_sentence(raw_text)
      preprocessed_text = loaded_tokenizer.texts_to_sequences([normalized_text])
      preprocessed_text = pad_sequences(preprocessed_text, maxlen=MAXLEN,padding='post')

      prediction = label_encoder.inverse_transform(np.argmax(model.predict([preprocessed_text]), axis=-1))[0]

      class_probabilities = zip(label_encoder.classes_, model.predict([preprocessed_text])[0])


      with col1:
        st.success("Your Entered Tweets")
        st.write(raw_text)
        st.success("Preprocessed Tweets")
        st.write(f"{normalized_text}")
        st.success("Padded Tweets")
        st.write(f"{preprocessed_text}")

      with col2:
        st.success("Predicted Emotion")
        emoji_icon = emoji[prediction]
        st.write("{}:{}".format(prediction,emoji_icon))
      
        st.success("Distribution of Emotion Prediction Probability")
        # for label, probability in class_probabilities:
        #    labels, probabilities = zip(*class_probabilities)
        labels, probabilities = zip(*class_probabilities)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Plot the bar chart with labels and values
        fig, ax = plt.subplots()
        bars = ax.barh(labels, probabilities, color=colors)

        # Display probability values on the bars
        for bar, probability in zip(bars, probabilities):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f'{probability:.3f}', ha='center', va='center')

        ax.set_xlabel('Probability')
        ax.set_title('Emotion Prediction Probability Distribution')

        # Display the plot in Streamlit
        st.pyplot(fig)
  elif choice == "About":
    st.subheader("About This Project")
    # Display an image

    image_path = "MeetTheGroup.jpg"  # Replace with the actual path to your image file
    image_path = Image.open(image_path)
    st.image(image_path, caption="FYP Project Team", use_column_width=True)

  elif choice == "Data Exploration":
    st.subheader("Exploration of The Training Dataset")
    image_path = "emotiondistribution.jpg"  # Replace with the actual path to your image file
    image_path = Image.open(image_path)
    st.image(image_path, caption="Emotion Frequency Count Distribution", use_column_width=True)

  else:
    st.subheader("Findings")

if __name__ == '__main__':
  main()
    
