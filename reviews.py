import re
import string
import nltk
from nltk.corpus import stopwords
import csv
import pandas as pd
import os



def clean_text(text):
    # Convert text to lowercase
    text = str(text)
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Remove any additional whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text




def create_data_frame(file_name):
    csv_files = [file for file in os.listdir(f"/home/npastori/eecs298/review_pred/{file_name}") if file.endswith('.csv')]
    combined_data = pd.DataFrame()
    for file in csv_files:
        data = pd.read_csv(os.path.join(f'/home/npastori/eecs298/review_pred/{file_name}', file),usecols = ['reviews.rating','reviews.text'])
        combined_data = combined_data._append(data, ignore_index=True)

    combined_data['Ratings'] = combined_data['reviews.rating']
    combined_data['clean_text'] = combined_data['reviews.text'].apply(clean_text)

    useful_data = combined_data[['clean_text','Ratings']]
    return useful_data
