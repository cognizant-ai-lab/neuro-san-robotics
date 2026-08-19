#!/usr/bin/env python3
"""Self-signed TLS certificate management for the Flask UI.

Two entry points, one implementation:

* ``interface_flask.py`` imports :func:`ensure_certs` and calls it on every
  start, so the cert always matches the address the robot is currently on.
* Run directly as a CLI to inspect, or to force a refresh:
      python scripts/setup_tls_certs.py --check
      python scripts/setup_tls_certs.py --force

Why the app needs TLS: browsers only expose ``getUserMedia`` in a secure
context. Over plain HTTP the UI loads but the microphone silently fails from any
origin except localhost -- which is why an ``ssh -L`` tunnel works while direct
``http://<robot>:5001`` does not. Serving HTTPS makes direct access usable.

Why the cert is generated rather than committed: robots differ. Some hold a DHCP
lease that rotates, some are pinned to a fixed address. :func:`ensure_certs` is
idempotent, so one unconditional call at startup covers both -- a robot at a
stable address regenerates nothing after the first run, and a robot whose lease
moved gets a cert covering the new address automatically.

Environment:
    ROBOT_HOST_IP    Address to embed as an IP SAN (default: auto-detected)
    TLS_CERT_DIR     Directory holding cert.pem/key.pem (default: ~/certs)
    TLS_CERT_DAYS    Validity in days (default: 3000)
    TLS_EXTRA_SANS   Comma-separated extra SANs, e.g. "10.0.0.9,dog2.local"
    TLS_FORCE_REGEN  Set to 1 to regenerate on every start
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_DAYS = 3000

# Regenerate when the cert has less than this left, so a long-lived robot does
# not wake up one morning serving an expired cert.
EXPIRY_MARGIN = datetime.timedelta(days=30)


# ---------------------------------------------------------------------
# Paths and configuration (read per call so env changes are picked up)
# ---------------------------------------------------------------------

def cert_dir() -> Path:
    return Path(os.environ.get("TLS_CERT_DIR") or (Path.home() / "certs"))


def cert_path() -> Path:
    return cert_dir() / "cert.pem"


def key_path() -> Path:
    return cert_dir() / "key.pem"


def detect_host_ip() -> str | None:
    """Return this host's address on the default route, or None.

    Uses the default route rather than the hostname so it picks the interface
    other machines actually reach us on -- on these robots eth0 is the private
    link to the Go2 and must not end up in the cert as the UI address.
    """
    try:
        fields = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    if "src" in fields:
        return fields[fields.index("src") + 1]
    return None


def host_ip() -> str | None:
    """Configured UI address: ROBOT_HOST_IP if set, else auto-detected."""
    return (os.environ.get("ROBOT_HOST_IP") or "").strip() or detect_host_ip()


def build_sans(ip: str | None = None, extra: list[str] | None = None) -> list[str]:
    """Assemble the SAN list, most stable name first.

    The mDNS name leads because it survives an address change; the IP is a
    convenience that goes stale. Duplicates are dropped, order preserved.
    """
    if ip is None:
        ip = host_ip()
    if extra is None:
        raw = os.environ.get("TLS_EXTRA_SANS", "")
        extra = [item.strip() for item in raw.split(",") if item.strip()]

    hostname = socket.gethostname()
    sans = [f"DNS:{hostname}.local", f"DNS:{hostname}", "DNS:localhost"]
    if ip:
        sans.append(f"IP:{ip}")
    sans.append("IP:127.0.0.1")

    for item in extra:
        if item.startswith(("DNS:", "IP:")):
            sans.append(item)
        else:
            try:
                ipaddress.ip_address(item)
                sans.append(f"IP:{item}")
            except ValueError:
                sans.append(f"DNS:{item}")

    return list(dict.fromkeys(sans))


# ---------------------------------------------------------------------
# Inspecting what is on disk
# ---------------------------------------------------------------------

def read_cert_sans(cert: Path | None = None) -> set[str]:
    """Return the SANs present in the cert on disk, as DNS:/IP: strings."""
    cert = cert or cert_path()
    try:
        from cryptography import x509  # pylint: disable=import-outside-toplevel

        loaded = x509.load_pem_x509_certificate(cert.read_bytes())
        san = loaded.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        names = {f"DNS:{n}" for n in san.get_values_for_type(x509.DNSName)}
        addrs = {f"IP:{a}" for a in san.get_values_for_type(x509.IPAddress)}
        return names | addrs
    except Exception as exc:  # noqa: BLE001 - treated as "unreadable"
        logging.debug("Could not read SANs from %s: %s", cert, exc)
        return set()


def cert_expiry(cert: Path | None = None) -> datetime.datetime | None:
    """Return the cert's expiry as an aware UTC datetime, or None."""
    cert = cert or cert_path()
    try:
        from cryptography import x509  # pylint: disable=import-outside-toplevel

        return x509.load_pem_x509_certificate(cert.read_bytes()).not_valid_after_utc
    except Exception as exc:  # noqa: BLE001
        logging.debug("Could not read expiry from %s: %s", cert, exc)
        return None


def cert_status(wanted: list[str] | None = None) -> tuple[bool, str]:
    """Return (usable, human-readable reason) for the cert pair on disk."""
    if wanted is None:
        wanted = build_sans()

    if not cert_path().exists() or not key_path().exists():
        return False, "cert.pem/key.pem not present"

    missing = [san for san in wanted if san not in read_cert_sans()]
    if missing:
        return False, f"missing SANs: {', '.join(missing)}"

    expires = cert_expiry()
    if expires is None:
        return False, "cert unreadable"
    if expires - datetime.datetime.now(datetime.timezone.utc) < EXPIRY_MARGIN:
        return False, f"expires {expires:%Y-%m-%d}"

    return True, f"covers all SANs, valid until {expires:%Y-%m-%d}"


# ---------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------

def generate(sans: list[str] | None = None, days: int | None = None) -> None:
    """Write a fresh self-signed cert/key pair, replacing any existing one."""
    # pylint: disable=import-outside-toplevel
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    if sans is None:
        sans = build_sans()
    if days is None:
        days = int(os.environ.get("TLS_CERT_DAYS") or DEFAULT_DAYS)

    entries: list[x509.GeneralName] = []
    for san in sans:
        kind, _, value = san.partition(":")
        if kind == "IP":
            entries.append(x509.IPAddress(ipaddress.ip_address(value)))
        else:
            entries.append(x509.DNSName(value))

    # CN is legacy -- browsers read SANs only -- but set it to the stable name.
    common_name = sans[0].partition(":")[2] if sans else socket.gethostname()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_dir().mkdir(parents=True, exist_ok=True)

    key_path().write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path().chmod(0o600)

    cert_path().write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    cert_path().chmod(0o644)


def ensure_certs(force: bool | None = None) -> tuple[bool, str]:
    """Make sure a usable cert pair exists. Returns (regenerated, reason).

    Safe to call unconditionally on every app start: a no-op when the existing
    cert already covers the current address, regenerating when the lease moved.
    Never raises -- a cert problem should not stop the robot from booting, it
    should downgrade the UI to HTTP with a warning.
    """
    if force is None:
        force = os.environ.get("TLS_FORCE_REGEN", "").strip() in {"1", "true", "yes"}

    try:
        sans = build_sans()
        ok, reason = cert_status(sans)
        if ok and not force:
            return False, reason

        generate(sans)
        ok, reason = cert_status(sans)
        if not ok:
            logging.warning("TLS cert still unusable after regeneration: %s", reason)
        return True, reason
    except Exception as exc:  # noqa: BLE001 - never fatal
        logging.warning(
            "Could not prepare TLS certs (UI will fall back to HTTP): %s", exc
        )
        return False, f"error: {exc}"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or refresh the Flask UI's self-signed TLS cert.",
    )
    parser.add_argument(
        "--ip", help="IP to embed (default: $ROBOT_HOST_IP, else auto-detected)"
    )
    parser.add_argument(
        "--san", action="append", default=None, metavar="NAME",
        help="Extra SAN; repeatable. Accepts 'host', '10.0.0.5', or 'DNS:host'",
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help=f"Validity in days (default: $TLS_CERT_DAYS or {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if the current cert is still valid",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Report status, exit non-zero if a refresh is needed; change nothing",
    )
    args = parser.parse_args()

    ip = args.ip or host_ip()
    if not ip:
        print(
            "warning: no IP found (set ROBOT_HOST_IP or pass --ip); "
            "cert will cover hostnames only",
            file=sys.stderr,
        )

    sans = build_sans(ip=ip, extra=args.san)
    print(f"cert dir : {cert_dir()}")
    print(f"host ip  : {ip or '(none)'}")
    print(f"SANs     : {', '.join(sans)}")

    ok, reason = cert_status(sans)

    if args.check:
        print(f"status   : {'OK' if ok else 'NEEDS REFRESH'} -- {reason}")
        return 0 if ok else 1

    if ok and not args.force:
        print(f"status   : up to date -- {reason}")
        print("           (use --force to regenerate anyway)")
        return 0

    print(f"action   : regenerating -- {'forced' if args.force else reason}")
    try:
        generate(sans, days=args.days)
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not generate cert: {exc}", file=sys.stderr)
        return 1

    ok, reason = cert_status(sans)
    print(f"result   : {'OK' if ok else 'FAILED'} -- {reason}")
    print(f"           {cert_path()}")
    print(f"           {key_path()}")
    if not ok:
        return 1

    print()
    print("Restart interface_flask.py, then browse to:")
    print(f"    https://{socket.gethostname()}.local:5001")
    if ip:
        print(f"    https://{ip}:5001")
    return 0


if __name__ == "__main__":
    sys.exit(main())
