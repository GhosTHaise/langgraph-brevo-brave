from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from typing import Optional
from agent.configuration import Configuration
from langchain.chat_models.base import BaseChatModel

def init_model(config: Optional[RunnableConfig] = None) -> BaseChatModel:
    """Initialize the configured chat model."""
    configuration = Configuration.from_runnable_config(config)
    fully_specified_name = configuration.model
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)