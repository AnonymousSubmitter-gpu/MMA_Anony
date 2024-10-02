import os
import json
import sys
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import requests
import pandas as pd
#from IPython.display import Image
import base64
from langchain.chains import LLMChain
from langchain.prompts import BasePromptTemplate  # Assuming a base class exists
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
import argparse
from agent import VLAgent
from setting import Settings, get_all_model_settings, load_model_setting
from model import load_llm_from_config

import base64
from io import BytesIO

import torch
#from llava.model import LlavaLlamaForCausalLM
#from llava.model.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from transformers import AutoTokenizer, StoppingCriteria

def run(args):
    settings = Settings()
    settings.model = load_model_setting(args.model)
    print(settings.model)
    vllm = None
    try:
        print("load")
        vllm = load_llm_from_config(settings.model.llm)
        print("end load")
    except Exception as e:
        return f"LLM initialization check failed: {e}" 
    print("Initialization of vllm completed.")
    print(type(vllm))
    print(vllm)
    agent = VLAgent(llm = vllm)

    image_path_prefix = ".dataset/" #path for dataset 
    data_meta_csv_path = "running.csv" #path for csv file
   
   
    df = pd.read_csv(data_meta_csv_path)
    df[args.model] = ""
    import time
    count = 0
    for i, row in df.iterrows():
        start_time = time.time()
        image_path = image_path_prefix + row['file_name']
        image_path = os.path.abspath(image_path)
        
        question = row['question'] + " " + row['choice'] + "A: " + row['A'] + ' | B: ' + row['B'] + ' | C: ' + row['C'] + ' | D: ' + row['D'] + " Answer in format, like A | explanation."
        final_decision = agent.raw_inference(question, image_path)
        formatted_time = "{:.2f}".format(time.time() - start_time)
        print("final_answer = ", final_decision, "row['answer'] = ", row['answer'])
        
        df.at[i, args.model] = final_decision if final_decision else "No Response"
        print(f"make act total time cost: {formatted_time} s")

    df.to_csv(f"updated_merge_{args.model}_0605.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='VQA benchmark',
        description='improve multimodal inference ability',
        epilog='Text at the bottom of help')
    parser.add_argument("--seed", type=int, default=1, help="random_seed")
    parser.add_argument("--model", default="gemini-pro-vision", help="environment flag, openai-gpt-4-0613 or openai-gpt-3.5-turbo")
    args = parser.parse_args()
    print("seed = ", args.seed)
    print("model = ", args.model)
    run(args)
