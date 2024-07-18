# IMPORTS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import math
import openai
import re

# LANGCHAIN
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# ML
from sklearn.metrics import log_loss, accuracy_score
import torch 
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# GLOBALS
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI API KEY OBTAINED" if openai.api_key else "MISSING API KEY")

class ProcessData:
    def __init__(self, dataframe):
        self.DataFrame = dataframe
    
    def split_df(self):
        df = self.DataFrame.copy()
        
        def split_text(text):
            match = re.match(r"Interviewer:\s*(.*?)\s*Interviewee:\s*(.*)", text, re.DOTALL)
            if match:
                return match.groups()
            else:
                return None, None
        
        df[['Question', 'Response']] = df['Input Text'].apply(lambda x: pd.Series(split_text(x)))
        
        self.DataFrame = df
        return df

class Tagger:
    def __init__(self):
        pass
    
    # def set_dataframe(self, new_df):
    #     self.DataFrame = new_df.copy()
    #     return self.DataFrame

    def simple(self, question, response):
        tagging_prompt = ChatPromptTemplate.from_template(
            """
        Extract the desired information from the following passage.

        Only extract the properties mentioned in the 'Classification' function.

        Interviewer's Question:
        {question}

        Interviewee's Response:
        {response}
        """
        )

        options = {
            "description": """Rate the technical expertise of the interviewee response from 1 to 5 based on the rubric below. Ignore the quality of the interview, give the rating on technical merit alone. Keep in mind that it is not common for a response to be scored a 3, rare for a response to be scored a 4, and even more rare for a response to be scored a 5——only 16 percent of respondants were given a 3 or higher. The rubric is as follows:

        1 = Rarely uses technical terms and, when used, they are often inaccurate or misapplied. Lacks basic familiarity with relevant technologies, methodologies, or frameworks.

        2 = Employs some technical language but with frequent inaccuracies or in the wrong context. Knowledge of key technologies and concepts seems superficial based on examples given.

        3 = Technical terminology is generally used correctly, though some terms may be misused or the candidate may struggle to define terms they use. Demonstrates working knowledge of relevant methodologies/frameworks but examples don't showcase deep expertise. Level of technicality is not always well-calibrated.

        4 = Uses technical language accurately and can speak to technologies and frameworks in depth, though some niche terms or advanced concepts may not be fully mastered. Examples are insightful and mostly well-suited to the use case in terms of technical complexity.

        5 = Technical terms are used with flawless accuracy. Candidate not only meaningfully discusses key technologies, methodologies, and frameworks, but also chooses an optimal level of technicality for their given use case.""",
            "enum": [1, 2, 3, 4, 5]
        }
        description, enum = options["description"], options["enum"]


        class Classification(BaseModel):
            rating: int = Field(
                description=description,
                enum=enum,
            )

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").with_structured_output(
            Classification
        )

        chain = tagging_prompt | llm

        result = chain.invoke({"question": question, "response": response})
        
        return result.rating
    
    def apply_function_to_df(self, df, func, include_question=False, col_name="Prediction"):
        if include_question:
            df[col_name] = df.apply(lambda row: func(row["Question"], row["Response"]), axis=1)
        else:
            df[col_name] = df.apply(lambda row: func(row["Response"]), axis=1)
        return df
    
    def llm_valid_response(self, response):
        if isinstance(response, float) and math.isnan(response):
            return False
        elif len(response) < 50: # used to be 4, less than 10 words is unlikely to produce a favorable result 
            return False
        elif len(response) > 300:
            return True

        options = {
            "description": """If the response contains a request to repeat or clarify the question, return False. Return True otherwise. Do not judge the quality of the answer. Just judge whether the answer requests a clarification or not.""",
            "enum": [True, False]
        }
        return self.llm_bool(response, options)
        
    def llm_bool(self, response, options):
        description, enum = options["description"], options["enum"]

        tagging_prompt = ChatPromptTemplate.from_template(
            """
        You are given a response to a technical interview question. Extract the desired information from the following passage.

        Only extract the properties mentioned in the 'ValidResponse' function.

        Interviewee's Response:
        {response}
        """
        )
        
        class ValidResponse(BaseModel):
            rating: bool = Field(
                description=description,
                enum=enum,
            )

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").with_structured_output(
            ValidResponse
        )

        chain = tagging_prompt | llm

        result = chain.invoke({"response": response})
        
        return result.rating
    
    def xgb(self, train_data, eval_data):
        X_train = train_data['Response']
        y_train = train_data['Label']
        y_train = y_train - 1
        
        X_test = eval_data['Response']
        
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        
        xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_classifier.fit(X_train_tfidf, y_train)
        
        eval_data["xgb"] = xgb_classifier.predict(X_test_tfidf) + 1
        train_data["xgb"] = xgb_classifier.predict(X_train_tfidf) + 1
        
        class_counts = eval_data["xgb"].value_counts()
        print("XGB class counts for evaluation set:")
        print(class_counts)
        
        return eval_data, train_data

    def length_lr(self, train_data, eval_data):
        X_train = train_data['Response'].str.len().values.reshape(-1, 1)  
        y_train = train_data['Label'] 
        X_test = eval_data['Response'].str.len().values.reshape(-1, 1) 

        model = LogisticRegression(max_iter=1000)  # Increase the number of iterations
        model.fit(X_train, y_train)

        eval_data["length_lr"] = model.predict(X_test)
        train_data["length_lr"] = model.predict(X_train)
        
        class_counts = eval_data["length_lr"].value_counts()
        print("LENGTH LR class counts for evaluation set:")
        print(class_counts)
        
        return eval_data, train_data
    
    def super_tagger(self, train_data, eval_data):
        
        # Prepare training data
        X_train = train_data[["zero_shot", "length_lr", "xgb"]]
        y_train = train_data["Label"]

        # Prepare validation data
        X_val = eval_data[["zero_shot", "length_lr", "xgb"]]
        y_val = eval_data["Label"]

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_predictions)
        eval_data["pred_rf"] = rf_predictions
        print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

        # Train XGBoost
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train - 1)
        xgb_predictions = xgb_model.predict(X_val) + 1
        xgb_accuracy = accuracy_score(y_val, xgb_predictions)
        eval_data["pred_xgb"] = xgb_predictions
        print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

        # Train SVM
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_val)
        svm_accuracy = accuracy_score(y_val, svm_predictions)
        eval_data["pred_svm"] = svm_predictions
        print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
        
        return eval_data, train_data


class Evaluations:
    def __init__(self):
        pass
    
    def misclassified(self, df=None):
        if df is None:
            df = self.DataFrame
        
        return df[df["Label"] != df["Prediction"]]
    
    def evaluate(self, df=None, predict_col="Prediction", loss_type="accuracy"):
        # evaluate given that prediction is done
        if df is None:
            df = self.DataFrame
            
        if "Label" not in df.columns or predict_col not in df.columns:
            raise ValueError("DataFrame must have both 'Label' and 'Prediction' columns")
        
        true_labels = df["Label"]
        predicted_labels = df[predict_col]
        
        if loss_type == "ce":
            num_classes = np.unique(true_labels).size

            true_labels_one_hot = np.eye(num_classes)[true_labels - 1]
            predicted_labels_one_hot = np.eye(num_classes)[predicted_labels - 1]

            true_labels_tensor = torch.tensor(true_labels_one_hot, dtype=torch.float32)
            predicted_labels_tensor = torch.tensor(predicted_labels_one_hot, dtype=torch.float32)

            loss = F.cross_entropy(predicted_labels_tensor, true_labels_tensor)
        elif loss_type == "accuracy":
            loss = (predicted_labels == true_labels).mean() * 100
        else:
            raise ValueError("Invalid loss type")
        
        return loss
    

def main():
    train = pd.read_csv('../data/train.csv')
    # data = train.sample(frac=1).reset_index(drop=True)
    small = train
    small = pd.DataFrame(small)
    processor = ProcessData(small)
    baby = processor.split_df()
    
    tagger = Tagger()  
    df = tagger.apply_function_to_df(baby, tagger.simple)
    
    print(df.head())
    
    df.to_csv('../data/baseline.csv')

    loss = tagger.evaluate(df)
    print(f"Log Loss: {loss}")

if __name__ == "__main__":
    main()
