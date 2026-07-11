from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_local_sandbox_launcher_uses_isolated_persistent_runtime():
    launcher = REPOSITORY_ROOT / "scripts" / "start_local_sandbox.sh"
    content = launcher.read_text()

    assert "/data/L202500291/sandbox" in content
    assert "--host 127.0.0.1" in content
    assert "setsid nohup runuser -u sandboxer" in content
    assert "runuser -u sandboxer" in content
    assert "id sandboxer" in content
    assert "install -m 0644" in content
    assert 'base_url="http://${host}:${port}"' in content
    assert '"$base_url/docs"' in content
    assert '"$endpoint"' in content
    assert "print(1 + 1)" in content
    assert "assert not Path" in content
    assert "-o sandboxer" not in content
    assert "chown sandboxer" not in content


def test_local_sandbox_profile_blocks_network_and_privileges():
    profile = REPOSITORY_ROOT / "sandbox" / "firejail" / "sandbox.profile"
    content = profile.read_text()

    assert "net none" in content
    assert "caps.drop all" in content
    assert "seccomp" in content
    assert "rlimit cpu 3" in content


def test_local_sandbox_stop_script_uses_persistent_pid_file():
    stop_script = REPOSITORY_ROOT / "scripts" / "stop_local_sandbox.sh"
    content = stop_script.read_text()

    assert "/data/L202500291/sandbox/local-sandbox.pid" in content
    assert "kill" in content
