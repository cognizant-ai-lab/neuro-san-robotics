"""SAN selection for the UI's self-signed certificate."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import setup_tls_certs as tls


class CertSanTests(unittest.TestCase):
    """
    The cert has to cover wherever the robot actually is.

    ROBOT_HOST_IP is captured once when setmyenv.sh is sourced and then sticks
    for the life of that shell, so it goes stale the moment a robot changes
    network. Building the SAN list from it alone produced a cert for an address
    the robot no longer had, and cert_status() compared against that same stale
    value and called the cert fine -- so it never healed.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        patcher = patch.dict(os.environ, {
            "TLS_CERT_DIR": self._dir.name,
            "ROBOT_HOST_IP": "",
            "TLS_EXTRA_SANS": "",
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def sans_at(self, live, **env):
        """The SAN list this host would build while sitting at `live`."""
        with patch.dict(os.environ, env):
            with patch.object(tls, "detect_host_ip", return_value=live):
                return tls.build_sans()

    def status_at(self, live, **env):
        """Whether the cert on disk satisfies this host at `live`."""
        with patch.dict(os.environ, env):
            with patch.object(tls, "detect_host_ip", return_value=live):
                return tls.cert_status(tls.build_sans())[0]

    def generate_at(self, live, include_existing=True, **env):
        with patch.dict(os.environ, env):
            with patch.object(tls, "detect_host_ip", return_value=live):
                tls.generate(tls.build_sans(include_existing=include_existing))

    def ips(self):
        return {san for san in tls.read_cert_sans() if san.startswith("IP:")}

    # -- the case that must not change ---------------------------------

    def test_a_robot_that_has_not_moved_does_not_regenerate(self):
        """The upgrade case for robots already in the field."""
        self.generate_at("10.1.1.5", include_existing=False,
                         ROBOT_HOST_IP="10.1.1.5")
        self.assertTrue(self.status_at("10.1.1.5", ROBOT_HOST_IP="10.1.1.5"))

    def test_the_configured_address_is_still_covered(self):
        sans = self.sans_at("10.1.1.5", ROBOT_HOST_IP="10.1.1.5")
        self.assertIn("IP:10.1.1.5", sans)
        self.assertIn("IP:127.0.0.1", sans)
        self.assertTrue(any(s.endswith(".local") for s in sans))

    # -- the bug ------------------------------------------------------

    def test_a_stale_configured_address_no_longer_hides_the_real_one(self):
        """ROBOT_HOST_IP says one thing; the robot is somewhere else."""
        sans = self.sans_at("172.20.10.5", ROBOT_HOST_IP="172.16.6.176")
        self.assertIn("IP:172.16.6.176", sans, "configured address kept")
        self.assertIn("IP:172.20.10.5", sans, "live address must be covered too")

    def test_moving_network_is_noticed(self):
        self.generate_at("172.16.6.176", ROBOT_HOST_IP="172.16.6.176")
        self.assertFalse(
            self.status_at("172.20.10.5", ROBOT_HOST_IP="172.16.6.176"),
            "a cert missing the live address must not report itself fine",
        )

    # -- accumulation --------------------------------------------------

    def test_known_networks_are_remembered(self):
        """Moving between a few routers should settle, not churn."""
        for ip in ("10.0.1.9", "10.0.2.9", "10.0.3.9"):
            self.generate_at(ip)
        for ip in ("10.0.1.9", "10.0.2.9", "10.0.3.9"):
            self.assertTrue(self.status_at(ip), f"{ip} should already be covered")

    def test_a_brand_new_network_still_regenerates(self):
        self.generate_at("10.0.1.9")
        self.assertFalse(self.status_at("10.0.9.9"))

    def test_remembered_addresses_are_bounded(self):
        """A lease that rotates daily must not grow the cert without limit."""
        for index in range(tls.MAX_REMEMBERED_IPS + 8):
            self.generate_at(f"10.0.0.{index}")
        self.assertLessEqual(len(self.ips()), tls.MAX_REMEMBERED_IPS)
        self.assertIn("IP:10.0.0.%d" % (tls.MAX_REMEMBERED_IPS + 7), self.ips(),
                      "the current address must survive the cap")

    def test_names_are_never_capped(self):
        for index in range(tls.MAX_REMEMBERED_IPS + 8):
            self.generate_at(f"10.0.0.{index}")
        names = {san for san in tls.read_cert_sans() if san.startswith("DNS:")}
        self.assertIn("DNS:localhost", names)
        self.assertTrue(any(n.endswith(".local") for n in names))

    def test_fresh_drops_the_history(self):
        self.generate_at("10.0.1.9")
        self.generate_at("10.0.2.9")
        self.assertIn("IP:10.0.1.9", self.ips())
        self.generate_at("10.0.3.9", include_existing=False)
        self.assertNotIn("IP:10.0.1.9", self.ips())
        self.assertIn("IP:10.0.3.9", self.ips())

    # -- resilience ----------------------------------------------------

    def test_no_cert_on_disk_is_not_an_error(self):
        self.assertTrue(self.sans_at("10.1.1.5"))

    def test_an_unreadable_cert_is_not_an_error(self):
        Path(self._dir.name, "cert.pem").write_text("not a certificate")
        self.assertIn("IP:10.1.1.5", self.sans_at("10.1.1.5"))

    def test_a_host_with_no_route_still_builds_a_usable_list(self):
        """detect_host_ip() returns None off-network; names still work."""
        sans = self.sans_at(None)
        self.assertIn("IP:127.0.0.1", sans)
        self.assertTrue(any(s.startswith("DNS:") for s in sans))


if __name__ == "__main__":
    unittest.main()
