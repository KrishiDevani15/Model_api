from fastapi import FastAPI,Body
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from starlette import status
from typing import Dict, List
import os
from dotenv import load_dotenv
load_dotenv()
# 1) Create Api
app = FastAPI()

models_dict: Dict[str, List[str]] = {
    "OpenAi": ["gpt-4o", "gpt-4o-mini"],
    "DeepSeek ": ["deepseek-r1-distill-llama-70b"],
    "Google": ["gemma2-9b-it"],
    "Meta": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "Mistral": ["mixtral-8x7b-32768"],
}

@app.get("/get-models", response_model=Dict[str, List[str]])
async def get_models():
    return models_dict

class QueryBody(BaseModel):
    model_api_key:str
    model_type:str | None = None
    model_name:str
    query: str
    temperature_value:float= Field(ge=0,le=2)

@app.post("/response/",status_code=status.HTTP_200_OK)
async def reponse(model_info:QueryBody = Body(...)):
    model_type = model_info.model_type
    if model_type == "OpenAi":
        prompt = model_info
        if prompt:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot. Your name is Atom GPT,you always give the best."),
                ("user", """Answer the following Question: {user_prompt}.""")
            ])

            llm = ChatOpenAI(
                api_key = model_info.model_api_key, 
                model=model_info.model_name,
                temperature=model_info.temperature_value
            )
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"user_prompt":prompt})
            return{
                "Model_type":model_info.model_type,
                "query":model_info.query,
                "response":response
            }


    else:

        prompt = model_info
        if prompt:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI bot. Your name is Atom GPT."),
                ("user", """Answer the following Question: {user_prompt}.""")
            ])

            llm = ChatGroq(
                groq_api_key = model_info.model_api_key,
                model=model_info.model_name,
                temperature=model_info.temperature_value
            )
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"user_prompt":prompt})
            return{
                "Model_type":model_info.model_type,
                "query":model_info.query,
                "response":response
            }
