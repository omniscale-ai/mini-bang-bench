"""Real supervisor ownership regression under its exact four Linux capabilities."""
import json
import os
from pathlib import Path
import pwd
import secrets
import subprocess
import sys
import tarfile
import tempfile

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/mini_bang/pool/offline_service.py"
CAPABILITIES = (1 << 0) | (1 << 5) | (1 << 6) | (1 << 7)  # CHOWN, KILL, SETGID, SETUID


def _unused_uid():
    used = set()
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in path.read_text().splitlines():
                if line.startswith("Uid:"):
                    used.update(map(int, line.split()[1:]))
        except FileNotFoundError:
            pass
    for _ in range(100):
        uid = 2_000_000_000 + secrets.randbelow(1_000_000)
        if uid in used:
            continue
        try:
            pwd.getpwuid(uid)
        except KeyError:
            return uid
    raise AssertionError("No unused isolated high UID was found")


def _restricted_supervisor(source, spec_path):
    """Executed only in this test's child; never alter pytest's capabilities."""
    import ctypes
    import importlib.util
    module_spec = importlib.util.spec_from_file_location("permission_probe_supervisor", source)
    service = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(service)
    # No host-wide scan or kill is permitted in this regression. The tiny agent
    # has no descendants and supervise's subprocess.run waits for its sole child.
    stopped = []
    service.stop_processes = lambda uid, *args, **kwargs: stopped.append(uid)
    libc = ctypes.CDLL(None, use_errno=True)
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if int(status["CapPrm"], 16) & CAPABILITIES != CAPABILITIES:
        print("SKIP: root process lacks one of CHOWN/KILL/SETGID/SETUID")
        return 77
    bounding = int(status["CapBnd"], 16)
    if bounding & ~CAPABILITIES and not int(status["CapEff"], 16) & (1 << 8):
        print("SKIP: CAP_SETPCAP is needed only to reduce this child's bounding set")
        return 77
    assert libc.prctl(47, 4, 0, 0, 0) == 0  # clear ambient capabilities
    last = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    for capability in range(last + 1):
        if bounding & (1 << capability) and not CAPABILITIES & (1 << capability):
            assert libc.prctl(24, capability, 0, 0, 0) == 0  # drop bounding bit
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
    header, data = Header(0x20080522, 0), (Data * 2)()
    data[0].effective = data[0].permitted = CAPABILITIES
    assert libc.capset(ctypes.byref(header), ctypes.byref(data)) == 0
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    for key in ("CapEff", "CapPrm", "CapBnd"):
        assert int(status[key], 16) == CAPABILITIES
    assert all(int(status[key], 16) == 0 for key in ("CapInh", "CapAmb"))
    service.supervise(spec_path)
    assert all(uid == service.read(spec_path)["uid"] for uid in stopped)
    print(json.dumps({"supervisor_capabilities": CAPABILITIES, "host_cleanup_disabled": True,
                      "host_cleanup_calls": len(stopped)}))
    return 0


@pytest.mark.skipif(os.geteuid() != 0 or sys.platform != "linux", reason="Exact capability/UID regression needs Linux root")
def test_agent_files_output_and_archive_survive_with_exact_four_capabilities():
    uid = _unused_uid()
    with tempfile.TemporaryDirectory(prefix="mini-bang-supervisor-four-capabilities-", dir="/tmp") as directory:
        base = Path(directory)
        base.chmod(0o755)  # Every ancestor is traversable by the isolated agent UID.
        control = base / "control";control.mkdir(mode=0o700)
        expected = json.dumps({"message":"actual agent output", "uid":uid}, sort_keys=True) + "\n"
        code = ("from pathlib import Path\nimport os\n"
                f"assert os.getuid() == {uid}\n"
                "status = Path('/proc/self/status').read_text()\n"
                "caps = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)\n"
                "assert all(int(caps[key], 16) == 0 for key in ('CapEff','CapPrm','CapInh','CapAmb'))\n"
                "Path('agent-result.bin').write_bytes(b'actual agent file bytes')\n"
                f"Path('output/output.json').write_text({expected!r})\n"
                "print('isolated agent completed', flush=True)\n")
        spec = control / "launch.json"
        spec.write_text(json.dumps({"role":"agent", "uid":uid, "gid":uid,
            "workspace_root":str(base / "workspace"), "command":[sys.executable, "agent.py"],
            "files":{"agent.py":code}, "env":{}, "policy":None, "timeout":20}, indent=2))
        probe = ("import runpy,sys; module=runpy.run_path(sys.argv[1]); "
                 "raise SystemExit(module['_restricted_supervisor'](sys.argv[2],sys.argv[3]))")
        result = subprocess.run([sys.executable, "-I", "-c", probe, str(Path(__file__).resolve()), str(SOURCE), str(spec)],
            capture_output=True, text=True, timeout=40, check=False)
        if result.returncode == 77:
            pytest.skip(result.stdout.strip())
        assert result.returncode == 0, result.stdout + result.stderr
        proof = json.loads(result.stdout)
        assert proof["supervisor_capabilities"] == CAPABILITIES and proof["host_cleanup_disabled"] is True
        final = json.loads((control / "result.json").read_text())
        assert "infrastructure_error" not in final, final
        execution = final["execution_result"]
        assert execution["exit_code"] == 0 and execution["timed_out"] is False
        assert proof["host_cleanup_calls"] == 2
        assert execution["output"] == json.loads(expected)
        task = base / "workspace/task"
        assert (task / "agent.py").read_text() == code
        assert (task / "agent-result.bin").read_bytes() == b"actual agent file bytes"
        assert (task / "output/output.json").read_bytes() == expected.encode()
        with tarfile.open(execution["metadata"]["archive_path"]) as archive:
            assert archive.extractfile("workspace/agent.py").read() == code.encode()
            assert archive.extractfile("workspace/agent-result.bin").read() == b"actual agent file bytes"
            assert archive.extractfile("workspace/output/output.json").read() == expected.encode()
