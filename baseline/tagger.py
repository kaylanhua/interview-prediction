# IMPORTS
import pandas as pd
import numpy as np
from datetime import datetime
import os
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
from sklearn.metrics import log_loss

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
            "description": """Rate the technical expertise of the interviewee response from 1 to 5 based on the rubric below. Ignore the quality of the interview, give the rating on technical merit alone. Keep in mind that it is rare for a reponse to be scored a 4 and even more rare for a response to be scored a 5——only 16 percent of respondants were given a 3 or higher. The rubric is as follows:

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
    
    def apply_function_to_df(self, df, func):
        df["Prediction"] = df.apply(lambda row: func(row["Question"], row["Response"]), axis=1)
        return df
    
    def evaluate(self, eval_dataset=None):
        # evaluate given that prediction is done
        if eval_dataset is None:
            eval_dataset = self.DataFrame
        
        y_true = eval_dataset["Label"]
        y_pred = eval_dataset["Prediction"]
        
        # CE loss
        loss = log_loss(y_true, y_pred, labels=[1, 2, 3, 4])
        return loss
    

def main():
    train = pd.read_csv('../data/train.csv')
    # data = train.sample(frac=1).reset_index(drop=True)
    small = train[:10]
    small = pd.DataFrame(small)
    processor = ProcessData(small)
    baby = processor.split_df()
    
    tagger = Tagger()  
    df = tagger.apply_function_to_df(baby, tagger.simple)
    
    print(df.head())

    # # Evaluate the results
    # loss = tagger.evaluate(df)
    # print(f"Log Loss: {loss}")

if __name__ == "__main__":
    main()
