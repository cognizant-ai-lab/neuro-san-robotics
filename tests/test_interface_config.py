import ast
import os
import unittest
from pathlib import Path


def _load_passive_turn_helpers():
    """Load only the passive-turn config helpers without importing Flask runtime."""
    source_path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "conscious_assistant"
        / "interface_flask.py"
    )
    parsed = ast.parse(source_path.read_text(), filename=str(source_path))
    helper_names = {"_env_flag", "_should_enable_passive_agent_turns"}
    helper_defs = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name="os")])] + helper_defs,
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["_should_enable_passive_agent_turns"]


def _interface_source_text() -> str:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "conscious_assistant"
        / "interface_flask.py"
    )
    return source_path.read_text()


class InterfaceConfigTests(unittest.TestCase):
    def setUp(self):
        self._env_names = [
            "CONSCIOUS_ENABLE_PASSIVE_AGENT_TURNS",
        ]
        self._old_env = {name: os.environ.get(name) for name in self._env_names}
        for name in self._env_names:
            os.environ.pop(name, None)
        self._should_enable_passive_agent_turns = _load_passive_turn_helpers()

    def tearDown(self):
        for name, value in self._old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_passive_agent_turns_default_on(self):
        self.assertTrue(self._should_enable_passive_agent_turns())

    def test_explicit_current_env_can_disable_passive_agent_turns(self):
        os.environ["CONSCIOUS_ENABLE_PASSIVE_AGENT_TURNS"] = "0"

        self.assertFalse(self._should_enable_passive_agent_turns())

    def test_explicit_current_env_can_enable_passive_agent_turns(self):
        os.environ["CONSCIOUS_ENABLE_PASSIVE_AGENT_TURNS"] = "1"

        self.assertTrue(self._should_enable_passive_agent_turns())

    def test_passive_agent_turn_interval_is_configurable(self):
        source = _interface_source_text()

        self.assertIn("CONSCIOUS_PASSIVE_AGENT_TURN_INTERVAL_SECONDS", source)
        self.assertIn("last_passive_agent_turn_at", source)

    def test_passive_observation_waits_for_first_interactive_turn(self):
        source = _interface_source_text()

        self.assertIn("interactive_turn_seen = False", source)
        self.assertIn("if not interactive_turn_seen:", source)
        self.assertIn("interactive_turn_seen = True", source)


if __name__ == "__main__":
    unittest.main()
