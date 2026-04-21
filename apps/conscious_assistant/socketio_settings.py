import os


def env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a safe fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def socketio_server_options() -> dict:
    """
    Return conservative Socket.IO server settings for the Flask app.

    The robot deployment uses Flask-SocketIO in ``threading`` mode on top of
    Werkzeug. Polling is the most reliable transport in that configuration, so
    transport upgrades are disabled by default and can be re-enabled explicitly
    with ``CONSCIOUS_SOCKETIO_ALLOW_UPGRADES=1`` when a deployment supports it.
    """
    allow_upgrades = env_flag("CONSCIOUS_SOCKETIO_ALLOW_UPGRADES", False)
    enable_debug_logs = env_flag("CONSCIOUS_SOCKETIO_DEBUG", False)
    return {
        "async_mode": "threading",
        "cors_allowed_origins": "*",
        "always_connect": True,
        "allow_upgrades": allow_upgrades,
        "logger": enable_debug_logs,
        "engineio_logger": enable_debug_logs,
    }


def socketio_client_transports() -> list[str]:
    """
    Return the browser transport list that matches the server policy.

    By default we pin the browser to long-polling, which avoids brittle
    websocket handshakes on the robot's Werkzeug-based deployment. If upgrades
    are explicitly enabled, the browser uses the Engine.IO default ordering of
    polling first and websocket second.
    """
    raw_value = os.environ.get("CONSCIOUS_SOCKETIO_TRANSPORTS")
    if raw_value:
        transports = [
            entry.strip()
            for entry in raw_value.split(",")
            if entry.strip()
        ]
        if transports:
            return transports

    if env_flag("CONSCIOUS_SOCKETIO_ALLOW_UPGRADES", False):
        return ["polling", "websocket"]

    return ["polling"]
