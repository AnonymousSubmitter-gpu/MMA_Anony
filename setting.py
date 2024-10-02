from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic.v1 import BaseModel, BaseSettings, Extra
import os



class LLMSettings(BaseModel):
    """
    LLM/ChatModel related settings
    """

    type: str = "chatopenai"

    class Config:
        extra = Extra.allow


class EmbeddingSettings(BaseModel):
    """
    Embedding related settings
    """

    type: str = "openaiembeddings"

    class Config:
        extra = Extra.allow


class ModelSettings(BaseModel):
    """
    Model related settings
    """

    type: str = ""
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()

    class Config:
        extra = Extra.allow


class Settings(BaseSettings):
    """
    Root settings
    """

    name: str = "default"
    model: ModelSettings = ModelSettings()

    class Config:
        env_prefix = "suspicionagent_"
        env_file_encoding = "utf-8"
        extra = Extra.allow

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                #json_config_settings_source,
                env_settings,
                file_secret_settings,
            )


# ---------------------------------------------------------------------------- #
#                             Preset configurations                            #
# ---------------------------------------------------------------------------- #
class OpenAIGPT4Settings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-0613", max_tokens=3000,temperature=0.1,  request_timeout=120)
    embedding = EmbeddingSettings(type="openaiembeddings")

class OpenAIGPT432kSettings(ModelSettings):
    # NOTE: GPT4 is in waitlist
    type = "openai-gpt-4-32k-0613"
    llm = LLMSettings(type="chatopenai", model="gpt-4-32k-0613", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class OpenaiGPTOmni(ModelSettings):
    type = "gpt-4o"
    llm = LLMSettings(type="chatopenai", model="gpt-4o", max_tokens=200)
    embedding = EmbeddingSettings(type="openaiembeddings")

class OpenaiGPTVision(ModelSettings):
    type = "gpt-4-vision-preview"
    llm = LLMSettings(type="chatopenai", model="gpt-4-vision-preview", max_tokens=4096)
    embedding = EmbeddingSettings(type="openaiembeddings")

class GeminiVision(ModelSettings):
    tyep = "gemini-pro-vision"
    llm = LLMSettings(type="chatgemini", model="gemini-pro-vision", max_output_tokens=4096)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class Gemini(ModelSettings):
    tyep = "gemini-pro"
    llm = LLMSettings(type="chatgemini", model="gemini-pro", max_output_tokens=4096)
    embedding = EmbeddingSettings(type="openaiembeddings")

class GeminiLatestVision(ModelSettings):
    tyep = "gemini-1.5-pro-latest"
    llm = LLMSettings(type="chatgemini", model="gemini-1.5-pro-latest")
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class ClaudeVision(ModelSettings):
    tyep = "claude-3-opus-20240229"
    llm = LLMSettings(type="chatclaude", model="claude-3-opus-20240229")
    embedding = EmbeddingSettings(type="openaiembeddings")

class VILA_40b_Settings(ModelSettings):
    type = "VILA-40b"
    llm = LLMSettings(type="VILA-40b", model_name="Efficient-Large-Model/VILA1.5-40b", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class VILA_13b_Settings(ModelSettings):
    type = "VILA-13b"
    llm = LLMSettings(type="VILA-13b", model_name="Efficient-Large-Model/VILA1.5-13b", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class VILA_8b_Settings(ModelSettings):
    type = "VILA-8b"
    llm = LLMSettings(type="VILA-8b", model_name="Efficient-Large-Model/Llama-3-VILA1.5-8B", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class VILA_3b_Settings(ModelSettings):
    type = "VILA-3b"
    llm = LLMSettings(type="VILA-3b", model_name="Efficient-Large-Model/VILA1.5-3b", max_tokens=2500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class llavav_next_Settings(ModelSettings):
    type = "llavav-next-34b"
    llm = LLMSettings(type="llavav-next", model_name="llava-hf/llava-v1.6-34b-hf", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class llavav_next_7b_Settings(ModelSettings):
    type = "llavav-next-7b"
    llm = LLMSettings(type="llavav-next-7b", model_name="llava-hf/llava-v1.6-mistral-7b-hf", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class llavav_next_13b_Settings(ModelSettings):
    type = "llavav-next-13b"
    llm = LLMSettings(type="llavav-next-13b", model_name="llava-hf/llava-v1.6-vicuna-13b-hf", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class MiniCPMV_Setting(ModelSettings):
    type = "MiniCPMV"
    llm = LLMSettings(type="MiniCPMV", model_name="openbmb/MiniCPM-Llama3-V-2_5", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class Intern_VL_1_5_Settings(ModelSettings):
    type = "Intern-VL-1-5"
    llm = LLMSettings(type="Intern-VL-1-5", model_name="OpenGVLab/InternVL-Chat-V1-5", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class Cogvlm2_Settings(ModelSettings):
    type = "Cogvlm2"
    llm = LLMSettings(type="Cogvlm2", model_name="THUDM/cogvlm2-llama3-chat-19B", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class DeepSeekVL_Settings(ModelSettings):
    type = "DeepSeekVL"
    llm = LLMSettings(type="DeepSeekVL", model_name="deepseek-ai/deepseek-vl-7b-chat", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")

class HPT_AIR_Settings(ModelSettings):
    type = "HPT_AIR"
    llm = LLMSettings(type="HPT_AIR", model_name="HyperGAI/HPT1_5-Air-Llama-3-8B-Instruct-multimodal", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")
    
class MiniGPT4v2_Settings(ModelSettings):
    type = "MiniGPT4v2"
    llm = LLMSettings(type="MiniGPT4v2", model_name="MiniGPT4v2", max_tokens=500)
    embedding = EmbeddingSettings(type="openaiembeddings")

#THUDM/chatglm3-6b
# ------------------------- Model settings registry ------------------------ #
model_setting_type_to_cls_dict: Dict[str, Type[ModelSettings]] = {
    "gemini-pro-vision": GeminiVision,
    "gpt4o": OpenaiGPTOmni,
    "gemini-pro": Gemini,
    "gpt-4-vision-preview": OpenaiGPTVision,
    "claude-3-vision": ClaudeVision,
    "gemini-1-5": GeminiLatestVision,
    "VILA-40b": VILA_40b_Settings,
    "VILA-13b": VILA_13b_Settings,
    "VILA-8b": VILA_8b_Settings,
    "VILA-3b": VILA_3b_Settings,
    "llavav-next-34b": llavav_next_Settings,
    "llavav-next-7b": llavav_next_7b_Settings,
    "llavav-next-13b": llavav_next_13b_Settings,
    "MiniCPMV": MiniCPMV_Setting,
    "Intern-VL-1-5": Intern_VL_1_5_Settings,
    "Cogvlm2": Cogvlm2_Settings,
    "DeepSeekVL": DeepSeekVL_Settings,
    "HPT_AIR": HPT_AIR_Settings,
    "MiniGPT4v2": MiniGPT4v2_Settings,
}


def load_model_setting(type: str) -> ModelSettings:
    if type not in model_setting_type_to_cls_dict:
        raise ValueError(f"Loading {type} setting not supported")

    cls = model_setting_type_to_cls_dict[type]
    return cls()


def get_all_model_settings() -> List[str]:
    """Get all supported Embeddings"""
    return list(model_setting_type_to_cls_dict.keys())

