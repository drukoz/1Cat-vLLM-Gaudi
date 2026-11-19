# SPDX-License-Identifier: Apache-2.0
"""Tests for the deb cache of the rental container entry (RUK-3239).

`apt-get update` is very expensive on the rental node and every run starts a fresh container, so
`install_toolchain` keeps the habanalabs-tools debs under `/models/apt-cache` (the data disk) and
installs from there on reruns without any update; the crash probe likewise tries
`apt-get install gdb` before an update. These tests assert on the script text.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
CRASH_PROBE = REPO_ROOT / "tools" / "flash_next" / "gdn_verify_crash_probe.sh"


def test_scripts_bash_syntax():
    for script in (SCRIPTS_DIR / "container_entry.sh", CRASH_PROBE):
        proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script}: {proc.stderr}"


def test_container_entry_deb_cache_literals():
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert 'APT_DEB_CACHE="/models/apt-cache/habanalabs-tools-$HABANA_TOOLS_VERSION"' in text
    assert "--download-only" in text
    assert 'Dir::Cache::Archives="$APT_DEB_CACHE"' in text
    assert '"$APT_DEB_CACHE"/*.deb' in text
    assert "$APT_DEB_CACHE/complete" in text
    # The cache install is announced, and a broken cache is demarked for the refill path.
    assert "installing habanalabs-tools from the deb cache" in text
    assert "deb cache install failed; falling back to apt-get update" in text


def test_container_entry_deb_cache_install_before_update():
    # The cache path must be tried before any apt-get update, so reruns on the same node skip it:
    # the first cache install command comes before the first update command in the script.
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert text.index('apt-get install -y --no-install-recommends "$APT_DEB_CACHE"/*.deb') < text.index(
        "apt-get update || true"
    )


def test_container_entry_keeps_the_apt_install_and_direct_deb_fallback():
    # The pre-existing install commands stay as the fallback for cache_ok=0 (no /models).
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert "apt-get install -y --no-install-recommends habanalabs-tools=1.24.1-482" in text
    assert "habanalabs-tools_1.24.1-482_amd64.deb" in text
    assert "/etc/ld.so.conf.d/habanatools.conf" in text


def test_crash_probe_installs_gdb_before_update():
    # The container's apt lists are usually warm: the probe tries the install first.
    text = CRASH_PROBE.read_text()
    assert "apt-get install -y -qq gdb" in text
    assert text.index("apt-get install -y -qq gdb") < text.index("apt-get update")
