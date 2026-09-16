"""
Picks OpenAI or Azure OpenAI for the robot's audio APIs.

neuro-san's llm_config covers the *agents* and nothing else: it only ever
builds LangChain chat models, and a CodedTool is handed args and sly_data, not
a client. TTS, Whisper and realtime transcription are HTTP calls this repo
makes itself, so they need their own provider switch. That is all this module
is.

It stays small because the openai SDK already reads AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_API_KEY and OPENAI_API_VERSION from the environment -- the same
variables neuro-san reads for the agents. So one Azure credential block serves
both, and the only thing left to supply is the deployment name, which on Azure
is what routes a request and rarely matches the model name.

Environment:
  GO2_AUDIO_PROVIDER               "auto" (default), "azure" or "openai".
                                   Set "openai" to keep audio on public OpenAI
                                   while the agents run on Azure -- Azure's
                                   audio models are region-limited and may not
                                   exist in the customer's chat resource.
  GO2_AZURE_TTS_DEPLOYMENT         deployment serving the TTS model
  GO2_AZURE_TRANSCRIBE_DEPLOYMENT  deployment serving Whisper
  GO2_AZURE_REALTIME_DEPLOYMENT    deployment serving realtime transcription
"""

import os
from typing import Any, Dict, Optional


# Azure's GA realtime surface mirrors OpenAI's path shape, so the two differ
# only in origin and in how the request is authenticated.
OPENAI_REALTIME_ROOT = "https://api.openai.com/v1/realtime"


def use_azure() -> bool:
    """Whether the audio APIs should talk to Azure rather than public OpenAI."""
    provider = os.environ.get("GO2_AUDIO_PROVIDER", "auto").strip().lower()
    if provider in {"azure", "openai"}:
        return provider == "azure"
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip())


def create_client(want_async: bool = False, **kwargs: Any) -> Any:
    """
    Build the audio client for the active provider.

    Both classes read their own credentials from the environment, so nothing is
    passed here beyond per-call options such as timeout.
    """
    try:
        import openai
    except ImportError as error:
        raise RuntimeError("openai package not installed. Run: pip install openai") from error

    if not use_azure():
        factory = openai.AsyncOpenAI if want_async else openai.OpenAI
        return factory(**kwargs)

    # The SDK raises a bare ValueError when this is missing. Say what to set,
    # because the caller's fallback would otherwise drop the robot to espeak
    # with nothing in the log explaining why.
    if not os.environ.get("OPENAI_API_VERSION", "").strip():
        raise RuntimeError(
            "Azure audio selected but OPENAI_API_VERSION is not set. It is not "
            "discoverable from Azure -- pick a version from Microsoft's API "
            'version list, for example "2024-10-21".'
        )

    factory = openai.AsyncAzureOpenAI if want_async else openai.AzureOpenAI
    return factory(**kwargs)


def deployment_for(env_var: str, model: str) -> str:
    """
    Return what belongs in a request's `model` field.

    Azure routes on the deployment name; falling back to the model name covers
    the common case of a deployment named after the model it serves.
    """
    if not use_azure():
        return model
    return os.environ.get(env_var, "").strip() or model


def realtime_urls() -> Dict[str, str]:
    """
    Return the realtime endpoints for the active provider.

    `client_secrets` is called by this server to mint a short-lived credential;
    `calls` is handed to the browser, which posts its WebRTC offer there.
    """
    if use_azure():
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
        if not endpoint:
            raise RuntimeError(
                "Azure audio selected but AZURE_OPENAI_ENDPOINT is not set."
            )
        root = f"{endpoint}/openai/v1/realtime"
    else:
        root = OPENAI_REALTIME_ROOT
    return {"client_secrets": f"{root}/client_secrets", "calls": f"{root}/calls"}


def realtime_api_key() -> Optional[str]:
    """Return the key used to mint a realtime session for the active provider."""
    name = "AZURE_OPENAI_API_KEY" if use_azure() else "OPENAI_API_KEY"
    return os.environ.get(name, "").strip() or os.environ.get("OPENAI_API_KEY", "").strip() or None


def realtime_auth_headers(api_key: str) -> Dict[str, str]:
    """
    Return the auth headers for a realtime session request.

    Azure takes a resource key in an `api-key` header; public OpenAI expects a
    bearer token. An Entra ID token is a bearer token on Azure too, so it is
    sent that way when one is configured.
    """
    if not use_azure():
        return {"Authorization": f"Bearer {api_key}"}
    entra_token = os.environ.get("AZURE_OPENAI_AD_TOKEN", "").strip()
    if entra_token:
        return {"Authorization": f"Bearer {entra_token}"}
    return {"api-key": api_key}
