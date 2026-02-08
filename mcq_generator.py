import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from groq import Groq

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0
)

def generate_mcq(text, num_questions=3):
    prompt = f"""
    Generate {num_questions} multiple choice questions from the text below.
    Each question must have 4 options and clearly indicate the correct answer.

    Text:
    {text}
    """

    response = llm.invoke(prompt)
    return response.content  
