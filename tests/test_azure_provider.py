"""Provider selection for the agents (HOCON) and the audio APIs (openai_provider)."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.conscious_assistant import interface_flask
from apps.conscious_assistant import realtime_transcription
from coded_tools.unigo2 import openai_provider
from coded_tools.unigo2 import tts_go2


ROOT = Path(__file__).resolve().parents[1]

# Everything the two providers key off. Cleared before each case so a developer's
# own Azure settings cannot make these pass or fail by accident.
PROVIDER_VARS = (
    "OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_AD_TOKEN", "OPENAI_API_VERSION", "GO2_AUDIO_PROVIDER",
    "GO2_AZURE_TTS_DEPLOYMENT", "GO2_AZURE_TRANSCRIBE_DEPLOYMENT",
    "GO2_AZURE_REALTIME_DEPLOYMENT", "AGENT_LLM_CLASS", "AGENT_LLM_MODEL_NAME",
    "AGENT_LLM_MODEL_NAME_LIGHT", "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_DEPLOYMENT_NAME_LIGHT",
)

AZURE_ENV = {
    "AZURE_OPENAI_ENDPOINT": "https://acme.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "azure-secret",
    "OPENAI_API_VERSION": "2024-10-21",
}


def provider_env(**overrides):
    """Return an environment with every provider variable under test control."""
    env = {name: "" for name in PROVIDER_VARS}
    env.update(overrides)
    return env


class AudioProviderTests(unittest.TestCase):
    """The audio APIs are configured separately from the agents."""

    def test_public_openai_is_the_default(self):
        with patch.dict(os.environ, provider_env(OPENAI_API_KEY="sk-test")):
            self.assertFalse(openai_provider.use_azure())
            self.assertEqual(type(openai_provider.create_client()).__name__, "OpenAI")
            self.assertEqual(
                openai_provider.realtime_urls()["client_secrets"],
                "https://api.openai.com/v1/realtime/client_secrets",
            )
            self.assertEqual(
                openai_provider.realtime_auth_headers("k"), {"Authorization": "Bearer k"}
            )

    def test_an_azure_endpoint_switches_the_audio_apis_over(self):
        with patch.dict(os.environ, provider_env(**AZURE_ENV)):
            self.assertTrue(openai_provider.use_azure())
            self.assertEqual(type(openai_provider.create_client()).__name__, "AzureOpenAI")
            self.assertEqual(
                type(openai_provider.create_client(want_async=True)).__name__,
                "AsyncAzureOpenAI",
            )
            urls = openai_provider.realtime_urls()
            self.assertEqual(
                urls["client_secrets"],
                "https://acme.openai.azure.com/openai/v1/realtime/client_secrets",
            )
            self.assertEqual(urls["calls"], "https://acme.openai.azure.com/openai/v1/realtime/calls")

    def test_azure_authenticates_with_an_api_key_header(self):
        """Azure takes a resource key in api-key; only Entra ID uses a bearer."""
        with patch.dict(os.environ, provider_env(**AZURE_ENV)):
            self.assertEqual(openai_provider.realtime_auth_headers("k"), {"api-key": "k"})
        with patch.dict(os.environ, provider_env(**AZURE_ENV, AZURE_OPENAI_AD_TOKEN="entra")):
            self.assertEqual(
                openai_provider.realtime_auth_headers("k"), {"Authorization": "Bearer entra"}
            )

    def test_audio_can_stay_on_public_openai_while_agents_use_azure(self):
        """Azure's audio models are region-limited, so the two must be separable."""
        env = provider_env(**AZURE_ENV, OPENAI_API_KEY="sk-test", GO2_AUDIO_PROVIDER="openai")
        with patch.dict(os.environ, env):
            self.assertFalse(openai_provider.use_azure())
            self.assertEqual(type(openai_provider.create_client()).__name__, "OpenAI")

    def test_azure_can_be_forced_without_an_endpoint_guess(self):
        with patch.dict(os.environ, provider_env(**AZURE_ENV, GO2_AUDIO_PROVIDER="azure")):
            self.assertTrue(openai_provider.use_azure())

    def test_deployment_name_replaces_the_model_only_on_azure(self):
        with patch.dict(os.environ, provider_env(OPENAI_API_KEY="sk-test")):
            self.assertEqual(
                openai_provider.deployment_for("GO2_AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts"),
                "gpt-4o-mini-tts",
            )
        with patch.dict(os.environ, provider_env(**AZURE_ENV, GO2_AZURE_TTS_DEPLOYMENT="acme-tts")):
            self.assertEqual(
                openai_provider.deployment_for("GO2_AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts"),
                "acme-tts",
            )
        # A deployment named after its model is the common case; fall back to it.
        with patch.dict(os.environ, provider_env(**AZURE_ENV)):
            self.assertEqual(
                openai_provider.deployment_for("GO2_AZURE_TTS_DEPLOYMENT", "gpt-4o-mini-tts"),
                "gpt-4o-mini-tts",
            )

    def test_missing_api_version_explains_itself(self):
        """The SDK raises a bare ValueError here, which reads as a TTS outage."""
        env = provider_env(
            AZURE_OPENAI_ENDPOINT="https://acme.openai.azure.com",
            AZURE_OPENAI_API_KEY="azure-secret",
        )
        with patch.dict(os.environ, env):
            with self.assertRaises(RuntimeError) as caught:
                openai_provider.create_client()
        self.assertIn("OPENAI_API_VERSION", str(caught.exception))


class LegacyTtsDeploymentTests(unittest.TestCase):
    """Some Azure regions offer only tts / tts-hd, which reject style guidance."""

    def test_style_guidance_is_sent_when_configured(self):
        self.assertEqual(
            tts_go2._style_kwargs("Speak warmly."), {"instructions": "Speak warmly."}
        )

    def test_style_guidance_is_dropped_when_blanked(self):
        """Blanking it must omit the parameter, not send an empty one."""
        for blank in ("", "   ", None):
            self.assertEqual(tts_go2._style_kwargs(blank), {}, repr(blank))


class RealtimeRoutingTests(unittest.TestCase):
    """The realtime session request has to follow the active provider."""

    def test_session_request_targets_azure_with_its_own_auth(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            raise OSError("stop after capture")

        with patch.dict(os.environ, provider_env(**AZURE_ENV)):
            with patch.object(realtime_transcription, "urlopen", fake_urlopen):
                with self.assertRaises(OSError):
                    realtime_transcription.request_realtime_session("azure-secret", "deploy")

        self.assertEqual(
            captured["url"],
            "https://acme.openai.azure.com/openai/v1/realtime/client_secrets",
        )
        headers = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(headers.get("Api-key".lower()), "azure-secret")
        self.assertNotIn("authorization", headers)

    def test_token_route_tells_the_browser_where_to_send_its_offer(self):
        success = realtime_transcription.RealtimeSessionResponse(
            200, "application/json", b'{"value":"ek_x"}', "req_ok",
        )
        with patch.dict(os.environ, provider_env(**AZURE_ENV)):
            with patch.object(interface_flask, "request_realtime_session", return_value=success):
                with interface_flask.app.test_client() as client:
                    response = client.post("/api/realtime/transcription-token")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["X-Realtime-Calls-Url"],
            "https://acme.openai.azure.com/openai/v1/realtime/calls",
        )
        # The credential still reaches the browser untouched.
        self.assertEqual(response.data, b'{"value":"ek_x"}')

    def test_browser_follows_that_header_instead_of_hardcoding_openai(self):
        browser = (ROOT / "apps" / "conscious_assistant" / "templates" / "index.html").read_text()
        self.assertIn("X-Realtime-Calls-Url", browser)
        self.assertIn("fetch(callsUrl", browser)


class AgentLlmConfigTests(unittest.TestCase):
    """registries/llm_config.hocon drives every agent network from the env."""

    def resolve(self, **env):
        """Parse both agent networks in a clean interpreter and return llm_config."""
        script = (
            "import json\n"
            "from pyhocon import ConfigFactory\n"
            "out = {}\n"
            "for n in ('conscious_agent', 'unigo2'):\n"
            "    out[n] = dict(ConfigFactory.parse_file('registries/%s.hocon' % n)['llm_config'])\n"
            "print(json.dumps(out))\n"
        )
        child = dict(os.environ)
        for name in PROVIDER_VARS:
            child.pop(name, None)
        child.update(env)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT), env=child, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        return json.loads(result.stdout)

    def test_default_is_public_openai_with_todays_models(self):
        config = self.resolve()
        self.assertEqual(
            config["conscious_agent"], {"class": "openai", "model_name": "gpt-5.1"}
        )
        # Pinned with its date because naming a class skips neuro-san's
        # catalogue, which is what used to expand a bare "gpt-4o" to this.
        self.assertEqual(
            config["unigo2"], {"class": "openai", "model_name": "gpt-4o-2024-08-06"}
        )

    def test_one_variable_moves_every_agent_network_to_azure(self):
        config = self.resolve(
            AGENT_LLM_CLASS="azure-openai",
            AZURE_OPENAI_DEPLOYMENT_NAME="acme-chat",
            AZURE_OPENAI_DEPLOYMENT_NAME_LIGHT="acme-chat",
            OPENAI_API_VERSION="2024-10-21",
        )
        for network in ("conscious_agent", "unigo2"):
            self.assertEqual(config[network]["class"], "azure-openai")
            self.assertEqual(config[network]["deployment_name"], "acme-chat")
            self.assertEqual(config[network]["openai_api_version"], "2024-10-21")

    def test_a_model_neuro_san_has_never_heard_of_still_works(self):
        """Customers deploy models newer than the installed neuro-san."""
        config = self.resolve(
            AGENT_LLM_CLASS="azure-openai",
            AGENT_LLM_MODEL_NAME="gpt-5.6",
            AZURE_OPENAI_DEPLOYMENT_NAME="acme-gpt56",
            AZURE_OPENAI_DEPLOYMENT_NAME_LIGHT="acme-mini",
            AGENT_LLM_MODEL_NAME_LIGHT="gpt-4o-mini",
            OPENAI_API_VERSION="2024-10-21",
        )
        self.assertEqual(config["conscious_agent"]["model_name"], "gpt-5.6")
        self.assertEqual(config["conscious_agent"]["deployment_name"], "acme-gpt56")
        self.assertEqual(config["unigo2"]["model_name"], "gpt-4o-mini")
        self.assertEqual(config["unigo2"]["deployment_name"], "acme-mini")

    def test_model_can_change_on_public_openai_too(self):
        config = self.resolve(AGENT_LLM_MODEL_NAME="gpt-5.2")
        self.assertEqual(
            config["conscious_agent"], {"class": "openai", "model_name": "gpt-5.2"}
        )

    def test_no_null_deployment_leaks_when_azure_is_unused(self):
        """A null here would be handed to a provider that never asked for it."""
        for network, config in self.resolve().items():
            self.assertNotIn("deployment_name", config, network)
            self.assertNotIn("openai_api_version", config, network)


if __name__ == "__main__":
    unittest.main()
