import re
from datetime import datetime
from typing import List, Optional, Tuple
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from pydantic.v1 import BaseModel, Field
from termcolor import colored
#import util
import time
from typing import Any
from langchain_core.messages import HumanMessage
from model import LlavaNext #, VILA #MiniGPT4v2#, HPT_AIR, MiniCPMV,DeepSeekVL#,Llava #Cogvlm2, InternVL  #VILA #Llava, LlavaNext
#from llava.model.language_model.llava_llama import LlavaLlamaModel
import base64
import requests
import sys
from PIL import Image, ImageDraw
import ast
import os
import numpy as np

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class VLAgent(BaseModel):
    llm: Any
    output_dir: str="make_image"
    def set_output_directory(self, output_dir):
        """Sets the output directory for marked images."""
        self.output_dir = output_dir

    def raw_inference(self, question, image):
        input_str = f"{question}.".format(question) #Answer in this format: XX(noun) is YY(adjective). Or it suggests/imples the meaning of the sentence I am asking." #, better to use a statement to answer the question with those word who has same/similiar meaning of the word in question, and shorten your answer."
        #imageData = encode_image(image)
        #print(input_str)
        if isinstance(self.llm, InternVL) or isinstance(self.llm, LlavaNext): #isinstance(self.llm, Llava) or isinstance(self.llm, LlavaNext):
            print("this is LLava model")
            image_input = {"url": image}
        else:
            imageData = encode_image(image)
            image_input = {"url": f"data:image/png;base64,{imageData}"}
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": input_str,
                },
                {
                    "type": "image_url",
                    "image_url": image_input,
                }
            ]
        )
        result = None
        try:
            result = self.llm.invoke([message]).content
        except Exception as e:
            print(e)
        return result
