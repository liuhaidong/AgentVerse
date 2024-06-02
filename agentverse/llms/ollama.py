import logging
import json
import ast
import os
import numpy as np
from aiohttp import ClientSession
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import Field

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from agentverse.message import Message

from . import llm_registry, OLLAMA_LLMS,OLLAMA_LLMS_MAPPING
from .base import BaseChatModel, BaseModelArgs
from .utils.jsonrepair import JsonRepair
from .utils.llm_server_utils import get_llm_server_modelname


from openai import OpenAI, AsyncOpenAI
from openai import OpenAIError
import ollama

class OllamaChatArgs(BaseModelArgs):
    model: str = Field(default="llama3-instruct")
    max_tokens: int = Field(default=8129)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)

api_key = None
base_url = None
model_name = None
    
DEFAULT_ARGS = OllamaChatArgs()
DEFAULT_BASE_URL = OLLAMA_LLMS_MAPPING[DEFAULT_ARGS.model]["base_url"]
DEFAULT_CLIENT = OpenAI(api_key='OPENAI_API_KEY', base_url=DEFAULT_BASE_URL)
DEFAULT_CLIENT_ASYNC = AsyncOpenAI(api_key='OPENAI_API_KEY', base_url=DEFAULT_BASE_URL)

# To support your own local LLMs, register it here and add it into LOCAL_LLMS.
@llm_registry.register("llama3-instruct")
class OllamaChat(BaseChatModel):
    args: OllamaChatArgs = Field(default_factory=OllamaChatArgs)
    client_args: Optional[Dict] = Field(
        default={"api_key": api_key, "base_url": base_url}
    )


    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OllamaChatArgs()
        args = args.model_dump()
        client_args = {"api_key": api_key, "base_url": base_url}
        # check if api_key is an azure key
        
        if args["model"] in OLLAMA_LLMS_MAPPING:
            client_args["api_key"] = OLLAMA_LLMS_MAPPING[args["model"]]["api_key"]
            client_args["base_url"] = OLLAMA_LLMS_MAPPING[args["model"]]["base_url"]
            client_args["model_name"] = OLLAMA_LLMS_MAPPING[args["model"]]["model_name"]
            is_azure = False
        else:
            raise ValueError(
                f"Model {args['model']} not found in LOCAL_LLMS_MAPPING"
            )
        super().__init__(
            args=args, max_retry=max_retry, client_args=client_args, is_azure=is_azure
        )

    @classmethod
    def send_token_limit(self, model: str) -> int:
        send_token_limit_dict = {
            "llama3-instruct": 8192,
        }
        # Default to 4096 tokens if model is not in the dictionary
        return send_token_limit_dict[model] if model in send_token_limit_dict else 4096

    # @retry(
    #     stop=stop_after_attempt(20),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    #     retry=retry_if_exception_type(
    #         exception_types=(OpenAIError, json.decoder.JSONDecodeError, Exception)
    #     ),
    # )
    def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)

        openai_client = OpenAI(
            api_key=self.client_args["api_key"],
            base_url=self.client_args["base_url"],
        )

        try:
            # Execute function call
            if functions != []:
                with openai_client:   
                    response = openai_client.chat.completions.create(
                        messages=messages,
                        functions=functions,
                        model=self.client_args["model_name"],
                    )

                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                if response.choices[0].message.function_call is not None:
                    self.collect_metrics(response)

                    return LLMResult(
                        content=response.choices[0].message.get("content", ""),
                        function_name=response.choices[0].message.function_call.name,
                        function_arguments=ast.literal_eval(
                            response.choices[0].message.function_call.arguments
                        ),
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                else:
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        content=response.choices[0].message.content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

            else:
                with openai_client:   
                    response = openai_client.chat.completions.create(
                        messages=messages,
                        model=self.client_args["model_name"],
                    )
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                self.collect_metrics(response)
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise

    # @retry(
    #     stop=stop_after_attempt(20),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    #     retry=retry_if_exception_type(
    #         exception_types=(OpenAIError, json.decoder.JSONDecodeError, Exception)
    #     ),
    # )
    async def agenerate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)


        async_openai_client = AsyncOpenAI(
            api_key=self.client_args["api_key"],
            base_url=self.client_args["base_url"],
        )
        try:
            if functions != []:
                response = await async_openai_client.chat.completions.create(
                    messages=messages,
                    functions=functions,
                    model=self.client_args["model_name"]
                )
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                if response.choices[0].message.function_call is not None:
                    function_name = response.choices[0].message.function_call.name
                    valid_function = False
                    if function_name.startswith("function."):
                        function_name = function_name.replace("function.", "")
                    elif function_name.startswith("functions."):
                        function_name = function_name.replace("functions.", "")
                    for function in functions:
                        if function["name"] == function_name:
                            valid_function = True
                            break
                    if not valid_function:
                        logger.warn(
                            f"The returned function name {function_name} is not in the list of valid functions. Retrying..."
                        )
                        raise ValueError(
                            f"The returned function name {function_name} is not in the list of valid functions."
                        )
                    try:
                        arguments = ast.literal_eval(
                            response.choices[0].message.function_call.arguments
                        )
                    except:
                        try:
                            arguments = ast.literal_eval(
                                JsonRepair(
                                    response.choices[0].message.function_call.arguments
                                ).repair()
                            )
                        except:
                            logger.warn(
                                "The returned argument in function call is not valid json. Retrying..."
                            )
                            raise ValueError(
                                "The returned argument in function call is not valid json."
                            )
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        function_name=function_name,
                        function_arguments=arguments,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                else:
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        content=response.choices[0].message.content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

            else:

                response = await async_openai_client.chat.completions.create(
                    messages=messages,
                    model=self.client_args["model_name"],
                )
                self.collect_metrics(response)
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise

    def construct_messages(
        self, prepend_prompt: str, history: List[dict], append_prompt: str
    ):
        messages = []
        if prepend_prompt != "":
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt != "":
            messages.append({"role": "user", "content": append_prompt})
        return messages

    def collect_metrics(self, response):
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens

    def get_spend(self) -> int:
        input_cost_map = {
            "llama3-instruct": 0.0001,
        }

        output_cost_map = {
            "llama3-instruct": 0.0001,
        }

        model = self.args.model
        if model not in input_cost_map or model not in output_cost_map:
            raise ValueError(f"Model type {model} not supported")

        return (
            self.total_prompt_tokens * input_cost_map[model] / 1000.0
            + self.total_completion_tokens * output_cost_map[model] / 1000.0
        )


def get_embedding(text: str, attempts=3) -> np.array:

    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    embedding = response["embedding"]
    return tuple(embedding)
