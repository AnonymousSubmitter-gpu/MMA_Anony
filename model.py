from typing import Dict, List, Type, Optional, Sequence

from PIL import Image
from langchain_community.chat_models import ChatOpenAI

from typing_extensions import TypeAlias
from langchain import chat_models, embeddings, llms
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLanguageModel
from setting import EmbeddingSettings, LLMSettings
from setting import Settings
from rich.console import Console
import torch.nn.functional as F
import gc
from pydantic import BaseModel, Extra, Field, root_validator
#from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AutoModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from langchain.schema.output import LLMResult
from typing import Any, Dict, List, Optional, Mapping, Tuple, Union
import os
from pydantic import BaseModel, Field, root_validator, Extra
from langchain_anthropic import ChatAnthropic
from typing import List, Optional, Dict, Any
import torch
from pydantic import root_validator
from langchain.schema.messages import AnyMessage, BaseMessage, get_buffer_string
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessageChunk,
)

from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
    PromptValue,
    RunInfo,
    Generation
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import sys
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#------
#------

def stable_softmax(logits):
    exps = torch.exp(logits - torch.max(logits))
    return exps / exps.sum(dim=-1, keepdim=True)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class LlavaNext(BaseLanguageModel):
    model_name: str
    llama_tokenizer: AutoTokenizer = Field(default=None)
    llama_model: LlavaNextForConditionalGeneration = Field(default=None)
    processor: LlavaNextProcessor = Field(default=None)
    
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        disable_torch_init()
        print("initilization")
        print(self.model_name)
        self.setup_model_and_tokenizer()


    
    def setup_model_and_tokenizer(self):
        print("come there to build")
        print("tokenizer name: ", self.model_name)

        self.processor = LlavaNextProcessor.from_pretrained(self.model_name) #("llava-hf/llava-v1.6-34b-hf")
        self.llama_model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True, device_map = "auto") #("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True, device_map = "auto")#.to("cuda") 
        self.llama_model.tie_weights()
        print("type(self.llama_tokenizer) = ",type(self.llama_tokenizer))
        print("type(self.llama_model) = ",type(self.llama_model))
        print("type(self.processor) = ",type(self.processor))

    def generate_text(self, input_ids):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llama_model = self.llama_model.to(device)
        sliced_input_ids = input_ids[0][:4500].unsqueeze(0)
        input_ids = sliced_input_ids.to(device)
        #input_ids = input_ids.to(device)
        gen_params = {
            'max_length': min(len(input_ids) * 2.5, 4500),
            'temperature': 0.9,
            'repetition_penalty': 1.0,
            'top_p': 0.7,
            'top_k': 50
        }
        output = self.llama_model.generate(input_ids, **gen_params)
        output_text = self.llama_tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def generate_prompt(self, prompts: List[str], 
        stop: Optional[List[str]] = None,
        **kwargs: Any,) -> LLMResult:
        # This method assumes that each prompt in the list will receive a separate response.
        # It returns a dictionary where the keys are the prompts and the values are the generated responses.
        responses = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        context_window = 22000

        input_ids = self.llama_tokenizer.encode(prompts.text, return_tensors="pt", max_length=context_window, truncation=True)

        sliced_input_ids = input_ids[0][:context_window].unsqueeze(0)

        input_ids = sliced_input_ids.to(device)

        attention_mask = [1] * len(input_ids[0])

        attention_mask = torch.tensor(attention_mask).to(device).unsqueeze(0)

        if "Based on the plan, please select the next action from the available action list" in prompt.text:
            print("make action, shrink the output length for valid output")
            max_l = int(2 * len(input_ids[0]))
        else:
            max_l = int(2 * len(input_ids[0]))
        print("len(input_ids[0]) = ", len(input_ids[0]))

        output = self.llama_model.generate(
            input_ids,
            attention_mask = attention_mask,
            pad_token_id= self.llama_tokenizer.eos_token_id, #self.llama_tokenizer.pad_token_id,
            max_length=min(context_window, max_l), #min(6000, 2 * len(input_ids[0])),  # or another value based on your needs
            temperature=0.7,
            repetition_penalty=1.0,
            no_repeat_ngram_size=5,
            top_p=0.7,
            top_k=50,
            do_sample=True,
        )
        output_text = self.llama_tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        if "Based on the plan, please select the next action from the available action list" in prompt.text:
            print("make action original output is: ", output_text)
        print(f"input len {len(input_ids[0])}, output len {len(output[0])}")
        gen = [[ChatGeneration(message=BaseMessage(content = output_text, type = "model_return"), generation_info=dict(finish_reason="stop"))]]
        llmoutput = {"token_usage": len(input_ids[0]) + len(output[0]), "model_name": self.model_name}
        responses = LLMResult(generations = gen, llm_output = llmoutput)
        return responses
    
    async def agenerate_prompt(self, prompts: List[str], *args, **kwargs) -> Dict[str, str]:
        # This method simply wraps the synchronous generate_prompt method for now,
        # as the provided code does not include asynchronous operations.
        return self.generate_prompt(prompts, *args, **kwargs)

    def invoke(self, input: str, *args, **kwargs) -> BaseMessage:
        qs = input[0].content[0]["text"] #input
        #print("read_image")
        image_file = input[0].content[1]["image_url"]['url']
        print(image_file)
        image = Image.open(image_file)
        if "7b" in self.model_name:
            print("7b model")
            prompt = f"[INST] <image>\n{qs}[/INST]".format(qs)
        elif "13b" in self.model_name:
            print("13b model")
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{qs} ASSISTANT:".format(qs)
        elif "34b" in self.model_name:
            print("34b model")
            prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{qs}<|im_end|><|im_start|>assistant\n".format(qs)
        else:
            raise("No such model")
        inputs = self.processor(prompt, image, return_tensors="pt")
        #print(inputs.keys())
        #print(inputs.shape)
        #print(len(inputs[0]))
        device = next(self.llama_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        #print("prompt = ", prompt)
        with torch.inference_mode():
            output_ids = self.llama_model.generate(
                **inputs,
                max_new_tokens=200,
            )
        outputs = self.processor.decode(output_ids[0], skip_special_tokens=True)

        outputs = outputs.strip()
        print("----------")
        #print("question:", args.question)
        print("outputs:", outputs)
        print("----------")
        
        #print(dir(input))
        
        
        result = BaseMessage(content = outputs, type = "model_return")
        return result


    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any) -> str:
        input_ids = self.llama_tokenizer.encode(text, return_tensors="pt")
        return self.generate_text(input_ids)

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        text = " ".join([message.content for message in messages])
        response_text = self.predict(text, stop=stop, **kwargs)
        return AIMessage(content=response_text)

    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        return self.predict(text, stop=stop, **kwargs)

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> BaseMessage:
        return self.predict_messages(messages, stop=stop, **kwargs)

    @property
    def InputType(self) -> TypeAlias:
        return str  # Assuming the input type is a string for this model.

    @property
    def _llm_type(self) -> str:
        return "Llama"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
        }



# ------------------------- LLM/Chat models registry ------------------------- #
llm_type_to_cls_dict: Dict[str, Type[BaseLanguageModel]] = {
    "chatopenai": ChatOpenAI,
    'chatgemini': ChatGoogleGenerativeAI,
    'chatclaude': ChatAnthropic,
    "llavav-next-34b": LlavaNext,
    "llavav-next-7b": LlavaNext,
    "llavav-next-13b": LlavaNext,
}


# ---------------------------------------------------------------------------- #
#                                LLM/Chat models                               #
# ---------------------------------------------------------------------------- #
def load_llm_from_config(config: LLMSettings) -> BaseLanguageModel:
    """Load LLM from Config."""
    config_dict = config.dict()
    config_type = config_dict.pop("type")
    if config_type not in llm_type_to_cls_dict:
        raise ValueError(f"Loading {config_type} LLM not supported")
    cls = llm_type_to_cls_dict[config_type]
    print("load_llm_from_config")
    return cls(**config_dict)


def get_all_llms() -> List[str]:
    """Get all supported LLMs"""
    return list(llm_type_to_cls_dict.keys())

