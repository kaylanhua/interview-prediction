import spacy
import pandas as pd 
import numpy as np


# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")

data = pd.read_csv("../data/train_valid_all.csv")

data = data[:10]


def ner(text):
    print("----- text: ", text)
    # Process the text
    doc = nlp(text)

    # Print the entities
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
        
    return doc.ents

data['NER_Entities'] = data['Response'].apply(ner)

# Save the NER Response, Label, and Prediction columns into a new CSV
output_columns = ['Response', 'NER_Entities', 'Label', 'Prediction']
output_data = data[output_columns]
output_data.to_csv('../data/ner_output.csv', index=False)
