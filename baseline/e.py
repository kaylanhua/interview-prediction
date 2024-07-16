# IMPORTS
import pandas as pd
import numpy as np
from datetime import datetime
import os
import openai
# from openai import OpenAI

# LANGCHAIN
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# GLOBALS
openai.organization = "org-raWgaVqCbuR9YlP1CIjclYHk" # Harvard
openai.api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI API KEY OBTAINED" if openai.api_key else "MISSING API KEY")

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

options = {
    "description": """Rate the technical expertise of the interviewee response from 1 to 5 based on the rubric below. Ignore the quality of the interview, give the rating on technical merit alone. Keep in mind that it is rare for a reponse to be scored a 4 and even more rare for a response to be scored a 5. The rubric is as follows:

1 = Rarely uses technical terms and, when used, they are often inaccurate or misapplied. Lacks basic familiarity with relevant technologies, methodologies, or frameworks.

2 = Employs some technical language but with frequent inaccuracies or in the wrong context. Knowledge of key technologies and concepts seems superficial based on examples given.

3 = Technical terminology is generally used correctly, though some terms may be misused or the candidate may struggle to define terms they use. Demonstrates working knowledge of relevant methodologies/frameworks but examples don't showcase deep expertise. Level of technicality is not always well-calibrated.

4 = Uses technical language accurately and can speak to technologies and frameworks in depth, though some niche terms or advanced concepts may not be fully mastered. Examples are insightful and mostly well-suited to the use case in terms of technical complexity.

5 = Technical terms are used with flawless accuracy. Candidate not only meaningfully discusses key technologies, methodologies, and frameworks, but also chooses an optimal level of technicality for their given use case.""",
    "enum": [1, 2, 3, 4, 5]
}
description, enum = options["description"], options["enum"]


class Classification(BaseModel):
    approval: int = Field(
        description=description,
        enum=enum,
    )

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").with_structured_output(
    Classification
)

chain = tagging_prompt | llm

text0 = "That's a very good question. So for dashboard what I did was I had in my complete if we had some we had a team for business intelligence. They used to work in on creating dashboards and Tableau as well as power bi so I had sit with them and ask them like if I give you just data of reviews, what do you want to see? I had talked with multiple people like what they want to see in a just in the reviews in kpis. Let's say I am developing a KP what they want to see what they wanted to know apart from it. I've been working for a client who was also working on reviews and I got some ideas over there. Like let's say showing them the how the product has evolved over here. So let's say we have a Samsung a series phone. So how the A52 a53 a 54 so how it has evolved over years. What do people are talking about it? When it was launched people were completely about camera. The next iteration people they didn't complain about cameras. They were complaining about something else or let's say, oh they had perfected the phone something like this. So they wanna show a line chart for this and then showing them top keywords to people are using so let's see people are using the word. battery a lot since it is a since it's a very good phone for have that has a amazing battery life of six or seven hours and people are talking about battery. I can just show them a chart or bar graph that showed that battery has a most expensive word similarly to English sentences. They told me something else like display or brightness. So brightens is very low for the phone. So people are talking Negatively about this brightness so we can have a bar chart that shows that brightness is on the top of negative words apart from the other. other complaints that people have and yeah, so that's it."

result = chain.invoke({"input": text0})
print(result.approval)
