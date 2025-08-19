from typing import Any

def create_llm_engine(model_string: str, use_cache: bool = False, is_multimodal: bool = True, **kwargs) -> Any:
    """
    Factory function to create appropriate LLM engine instance.
    """

    # 默认防重复参数（可被用户覆盖）
    default_penalties = {
        # OpenAI 风格
        "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
        "presence_penalty": kwargs.get("presence_penalty", 0.5),
        # 其他通用参数
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9),
    }

    # 合并到 kwargs（用户传的优先）
    for k, v in default_penalties.items():
        kwargs.setdefault(k, v)

    if "azure" in model_string:
        from .azure import ChatAzureOpenAI
        model_string = model_string.replace("azure-", "")
        return ChatAzureOpenAI(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif any(x in model_string for x in ["gpt", "o1", "o3", "o4"]):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "claude" in model_string:
        from .anthropic import ChatAnthropic
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("top_p", 0.9)
        return ChatAnthropic(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif any(x in model_string for x in ["deepseek-chat", "deepseek-reasoner"]):
        from .deepseek import ChatDeepseek
        kwargs.setdefault("repetition_penalty", 1.2)
        return ChatDeepseek(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "gemini" in model_string:
        from .gemini import ChatGemini
        kwargs.setdefault("temperature", 0.7)
        kwargs.setdefault("top_p", 0.9)
        return ChatGemini(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "grok" in model_string:
        from .xai import ChatGrok
        kwargs.setdefault("frequency_penalty", 0.5)
        kwargs.setdefault("presence_penalty", 0.5)
        return ChatGrok(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "vllm" in model_string:
        from .vllm import ChatVLLM
        model_string = model_string.replace("vllm-", "")
        kwargs.setdefault("repetition_penalty", 1.2)
        return ChatVLLM(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "litellm" in model_string:
        from .litellm import ChatLiteLLM
        model_string = model_string.replace("litellm-", "")
        kwargs.setdefault("frequency_penalty", 0.5)
        kwargs.setdefault("presence_penalty", 0.5)
        return ChatLiteLLM(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "together" in model_string:
        from .together import ChatTogether
        model_string = model_string.replace("together-", "")
        kwargs.setdefault("repetition_penalty", 1.2)
        return ChatTogether(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)

    elif "ollama" in model_string:
        from .ollama import ChatOllama
        model_string = model_string.replace("ollama-", "")
        kwargs.setdefault("repetition_penalty", 1.2)
        return ChatOllama(model_string=model_string, use_cache=use_cache, is_multimodal=is_multimodal, **kwargs)
        
    else:
        raise ValueError(
            f"Engine {model_string} not supported. "
            "If you are using Azure OpenAI models, please ensure the model string has the prefix 'azure-'. "
            "For Together models, use 'together-'. For VLLM models, use 'vllm-'. For LiteLLM models, use 'litellm-'. "
            "For Ollama models, use 'ollama-'. "
            "For other custom engines, you can edit the factory.py file and add its interface file. "
            "Your pull request will be warmly welcomed!"
        )