"""Adversarial submitted programs for the executable sandbox (policy and misconfiguration probes).

Each attack is a complete submitted program run as ``submission.py input.json
output.json``. It tries to cross one boundary and, when it can still write,
reports what it observed under ``attack_report``. No attack contains a kernel
exploit; anything that would succeed is harmless (a TCP connect, an empty file
that is removed again, sleeping processes inside a bounded loop).

``contained`` decides from the observed outcome whether the boundary held.
Every attack must also end as a failed submission: its output never matches a
task output schema. ``local`` attacks run under the local emulation
(``executable_sandbox_emulation.py``); the others depend on Pod-only controls
named in ``pod_controls`` and are verified by manifest/unit tests or in a
cluster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

PRELUDE = '''\
import errno, json, os, sys
INPUT, OUTPUT = sys.argv[1], sys.argv[2]
with open(INPUT) as _stream:
    CONTEXT = json.load(_stream)
MARKER = CONTEXT["marker"]

def report(**observed):
    with open(OUTPUT, "w") as stream:
        json.dump({"attack_report": observed}, stream)

'''


@dataclass
class Outcome:
    attack: str
    exit_code: int | None
    timed_out: bool
    violation: str | None
    output: dict[str, Any] | None
    submission_failed: bool
    leftover_processes: list[int] = field(default_factory=list)
    stdout_bytes: int = 0
    rlimits: dict[str, int] = field(default_factory=dict)

    @property
    def report(self) -> dict[str, Any]:
        value = (self.output or {}).get("attack_report")
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Attack:
    name: str
    goal: str
    source: str
    contained: Callable[[Outcome], bool]
    local: bool = True
    timeout_seconds: float = 30
    pod_controls: str = ""


def _nothing_reached(key: str) -> Callable[[Outcome], bool]:
    return lambda outcome: outcome.exit_code == 0 and outcome.report.get(key) == []


_TCP = '''
import socket
reached = []
for host, port in (("1.1.1.1", 443), ("8.8.8.8", 53), ("10.43.0.1", 443), ("10.96.0.1", 443)):
    try:
        socket.create_connection((host, port), timeout=2).close()
        reached.append(f"{host}:{port}")
    except OSError:
        pass
report(reached=reached)
'''

_UDP_DNS = '''
import socket
reached = []
try:
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"\\0" * 12, ("8.8.8.8", 53))
    reached.append("udp:8.8.8.8:53")
except OSError:
    pass
for name in ("example.com", "kubernetes.default.svc.cluster.local"):
    try:
        socket.getaddrinfo(name, 443)
        reached.append("dns:" + name)
    except OSError:
        pass
report(reached=reached)
'''

_METADATA = '''
import socket
reached = []
for port in (80, 443):
    try:
        socket.create_connection(("169.254.169.254", port), timeout=2).close()
        reached.append(port)
    except OSError:
        pass
report(reached=reached)
'''

_KUBERNETES = '''
import socket
host = os.environ.get("KUBERNETES_SERVICE_HOST")
reached = []
for target in [t for t in (host, "kubernetes.default.svc", "kubernetes.default") if t]:
    try:
        socket.create_connection((target, int(os.environ.get("KUBERNETES_SERVICE_PORT", 443))), timeout=2).close()
        reached.append(target)
    except OSError:
        pass
report(reached=reached, service_env=bool(host))
'''

_ENVIRONMENT = '''
words = ("TOKEN", "SECRET", "PASSWORD", "AWS", "KUBERNETES", "CREDENTIAL", "MINI_BANG_POOL", "MINI_BANG_RUNNER")
found = sorted(k for k in os.environ if any(w in k.upper() for w in words))
canary = [k for k, v in os.environ.items() if "canary-secret" in v]
report(reached=found + canary, keys=sorted(os.environ))
'''

_SERVICE_ACCOUNT = '''
reached = []
for path in ("/var/run/secrets/kubernetes.io/serviceaccount/token", "/run/secrets/kubernetes.io/serviceaccount/token",
             "/var/run/secrets/eks.amazonaws.com/serviceaccount/token"):
    try:
        with open(path, "rb") as stream:
            stream.read(1)  # Readability only; contents are never reported.
        reached.append(path)
    except OSError:
        pass
report(reached=reached)
'''

_PROCFS = '''
reached = []
me = os.getpid()
for pid in [p for p in os.listdir("/proc") if p.isdigit() and int(p) != me][:2000]:
    for name in ("environ", "mem"):
        try:
            with open(f"/proc/{pid}/{name}", "rb") as stream:
                if stream.read(1):
                    reached.append(f"{pid}/{name}")
        except OSError:
            pass
report(reached=reached)
'''

_SUPERVISOR = '''
import socket
reached = []
for path in ("/startup/offline-slot.json", "/startup/offline_service.py", "/control/offline.sock",
             "/control", "/mini-bang/workspace/monitor/execution-policy.json"):
    try:
        if os.path.isdir(path):
            os.listdir(path)
        elif path.endswith(".sock"):
            socket.socket(socket.AF_UNIX).connect(path)
        else:
            open(path, "rb").close()
        reached.append(path)
    except OSError:
        pass
report(reached=reached)
'''

_WRITE_OUTSIDE = '''
reached = []
for folder in ("/", "/etc", "/usr/lib", "/startup", "/mini-bang/checkpoints", "/control", os.path.dirname(INPUT)):
    path = os.path.join(folder, MARKER)
    try:
        with open(path, "x"):
            pass
        os.unlink(path)
        reached.append(folder)
    except OSError:
        pass
try:
    with open(INPUT, "a"):
        reached.append("input-manifest")
except OSError:
    pass
report(reached=reached)
'''

_FILE_SIZE = '''
written, error = 0, None
chunk = b"\\0" * (1024 * 1024)
with open(os.path.join(CONTEXT["workspace"], "fill.bin"), "wb", buffering=0) as stream:
    for _ in range(4096):  # bounded at 4 GiB even without any limit
        try:
            written += stream.write(chunk)
        except OSError as caught:
            error = errno.errorcode.get(caught.errno)
            break
report(written=written, error=error)
'''

_FORK_BOMB = '''
import time
created, error = 0, None
for _ in range(1000):  # bounded even if no process cap existed
    try:
        pid = os.fork()
    except OSError as caught:
        error = errno.errorcode.get(caught.errno)
        break
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    created += 1
report(created=created, error=error)
'''

_MEMORY = '''
import mmap
regions, error = [], None
for _ in range(1024):  # 64 GiB of address space at most; pages are never touched
    try:
        regions.append(mmap.mmap(-1, 64 * 1024 * 1024))
    except (OSError, MemoryError) as caught:
        error = type(caught).__name__
        break
report(reserved_bytes=len(regions) * 64 * 1024 * 1024, error=error)
'''

_DESCRIPTORS = '''
opened, error = [], None
for _ in range(100000):
    try:
        opened.append(os.open(os.devnull, os.O_RDONLY))
    except OSError as caught:
        error = errno.errorcode.get(caught.errno)
        break
for descriptor in opened:
    os.close(descriptor)
report(opened=len(opened), error=error)
'''

_THREADS = '''
import threading
threading.stack_size(256 * 1024)
gate, started, error = threading.Event(), 0, None
for _ in range(5000):
    try:
        threading.Thread(target=gate.wait, daemon=True).start()
        started += 1
    except RuntimeError as caught:
        error = str(caught)
        break
report(started=started, error=error)
gate.set()
'''

_SLEEP = '''
import time
time.sleep(3600)
report(survived=True)
'''

_SPIN = '''
while True:
    pass
'''

_SYMLINK = 'os.symlink("/etc/passwd", OUTPUT)\n'
_PROC_FD = 'os.symlink("/proc/self/fd/0", OUTPUT)\n'
_HARDLINK = '''
data = os.path.join(CONTEXT["workspace"], "features.json")
with open(data, "w") as stream:
    json.dump({"features": {"1": [[0.0]], "2": [[0.0]], "4": [[0.0]]}}, stream)
os.link(data, OUTPUT)
'''
_FIFO = 'os.mkfifo(OUTPUT)\n'
_DIRECTORY = 'os.mkdir(OUTPUT)\n'
_OVERSIZED = '''
with open(OUTPUT, "w") as stream:
    stream.write('{"padding": "' + "x" * (5 * 1024 * 1024) + '"}')
'''
_MALFORMED = '''
with open(OUTPUT, "w") as stream:
    stream.write('{"features": {"1": [[0.1, NaN')
'''
_NOT_OBJECT = '''
with open(OUTPUT, "w") as stream:
    json.dump([[0.1], [0.2]], stream)
'''
_FORGED = '''
envelope = {"exit_code": 0, "timed_out": False, "stderr": "",
            "metadata": {"outcome": "completed", "confirmed_participant_contract": True,
                         "sandbox": {"runtime_verified": True}, "policy_version": "forged"},
            "output": {"features": {"1": [[1.0]], "2": [[1.0]], "4": [[1.0]]}}}
with open(OUTPUT, "w") as stream:
    json.dump({"execution_result": envelope, **envelope}, stream)
'''

_DAEMON = '''
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        with open(os.path.join(CONTEXT["workspace"], "left-behind.txt"), "w") as stream:
            stream.write(MARKER)
        os.execv(sys.executable, [sys.executable, "-c", "import time\\nwhile True: time.sleep(1)", MARKER])
    os._exit(0)
os.wait()
report(daemon_started=True)
'''

_SIGNALS = '''
import signal
for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGUSR1):
    signal.signal(number, signal.SIG_IGN)
if os.fork() == 0:
    os.setsid()
    source = "import signal, time\\nfor s in (1, 2, 15): signal.signal(s, signal.SIG_IGN)\\nwhile True: time.sleep(1)"
    os.execv(sys.executable, [sys.executable, "-c", source, MARKER])
report(daemon_started=True)
'''

_PTRACE = '''
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
reached = []
for target in {os.getppid(), 1} - {os.getpid(), 0}:
    if libc.ptrace(16, target, 0, 0) == 0:  # PTRACE_ATTACH
        reached.append(target)
        libc.ptrace(17, target, 0, 0)  # PTRACE_DETACH
report(reached=reached)
'''

_NATIVE = '''
import ctypes, resource
import numpy as np
import scipy.fft, scipy.linalg
libc = ctypes.CDLL(None, use_errno=True)
ctypes_ok = libc.getpid() == os.getpid()
matrix = np.arange(16, dtype=float).reshape(4, 4) + np.eye(4)
numpy_ok = bool(np.isfinite(np.linalg.inv(matrix)).all())
scipy_ok = bool(abs(scipy.fft.fft(np.ones(8))[0] - 8) < 1e-9 and scipy.linalg.det(np.eye(3)) == 1.0)
raised = []
for name in ("RLIMIT_NPROC", "RLIMIT_AS", "RLIMIT_NOFILE", "RLIMIT_FSIZE", "RLIMIT_CORE"):
    kind = getattr(resource, name)
    soft, hard = resource.getrlimit(kind)
    try:
        resource.setrlimit(kind, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        raised.append(name)
    except (ValueError, OSError):
        pass
report(ctypes_ok=ctypes_ok, numpy_ok=numpy_ok, scipy_ok=scipy_ok, reached=raised)
'''

_INHERITED = '''
open_descriptors = sorted(int(name) for name in os.listdir("/proc/self/fd"))
report(reached=[d for d in open_descriptors if d > 3])
'''

_STDOUT_FLOOD = '''
line = "x" * 1023 + "\\n"
for _ in range(1024 * 1024):  # 1 GiB at most
    sys.stdout.write(line)
'''

def _limited(key: str, rlimit: str, *, errors: tuple[str | None, ...]) -> Callable[[Outcome], bool]:
    def check(outcome: Outcome) -> bool:
        value = outcome.report.get(key)
        return isinstance(value, int) and value <= outcome.rlimits[rlimit] and outcome.report.get("error") in errors
    return check


def _violation(*names: str) -> Callable[[Outcome], bool]:
    return lambda outcome: outcome.violation in names and outcome.output is None


def _no_survivors(outcome: Outcome) -> bool:
    return not outcome.leftover_processes


ATTACKS: tuple[Attack, ...] = (
    Attack("network-tcp-egress", "TCP to the internet and cluster service IPs", _TCP, _nothing_reached("reached"),
           pod_controls="NetworkPolicy deny ingress+egress; gVisor netstack"),
    Attack("network-udp-and-dns", "UDP datagram and DNS resolution", _UDP_DNS, _nothing_reached("reached"),
           pod_controls="egress deny; dnsPolicy None with nameserver 127.0.0.1"),
    Attack("network-cloud-metadata", "169.254.169.254 instance metadata", _METADATA, _nothing_reached("reached"),
           pod_controls="egress deny"),
    Attack("network-kubernetes-api", "Kubernetes API via service env/DNS", _KUBERNETES,
           lambda o: o.exit_code == 0 and o.report.get("reached") == [] and o.report.get("service_env") is False,
           pod_controls="enableServiceLinks false; no SA token; egress deny"),
    Attack("secrets-environment", "secrets in the child environment", _ENVIRONMENT, _nothing_reached("reached"),
           pod_controls="supervisor launches the child with a fixed minimal environment"),
    Attack("secrets-service-account-token", "service-account tokens", _SERVICE_ACCOUNT, _nothing_reached("reached"),
           local=False, pod_controls="automountServiceAccountToken false; secret/projected volumes refused"),
    Attack("procfs-other-processes", "/proc/<pid>/environ and mem of other processes", _PROCFS,
           _nothing_reached("reached"), local=False,
           pod_controls="participant UID 10001 vs root supervisor; ptrace-read check; gVisor procfs"),
    Attack("supervisor-files-and-socket", "startup sources, control socket, watchdog policy", _SUPERVISOR,
           _nothing_reached("reached"), local=False,
           pod_controls="startup ConfigMap 0400 root; /control 0700 root; SO_PEERCRED uid 0; monitor 0600 root"),
    Attack("write-outside-allowed-dirs", "write to /, /etc, inputs, control, input manifest", _WRITE_OUTSIDE,
           _nothing_reached("reached"), pod_controls="readOnlyRootFilesystem; read-only input PVC; root-owned 0444 inputs"),
    Attack("disk-fill-single-file", "one file larger than RLIMIT_FSIZE",
           _FILE_SIZE, _limited("written", "RLIMIT_FSIZE", errors=("EFBIG",)),
           pod_controls="RLIMIT_FSIZE; emptyDir sizeLimit evicts the one-shot Pod"),
    Attack("fork-bomb", "unbounded fork", _FORK_BOMB,
           lambda o: _limited("created", "RLIMIT_NPROC", errors=("EAGAIN",))(o) and _no_survivors(o),
           pod_controls="RLIMIT_NPROC in the child; cgroup pids.max or sandbox kernel"),
    Attack("memory-exhaustion", "reserve address space without bound", _MEMORY,
           lambda o: o.report.get("error") in ("OSError", "MemoryError")
           and o.report.get("reserved_bytes", 1 << 62) <= o.rlimits["RLIMIT_AS"],
           pod_controls="RLIMIT_AS; cgroup memory limit (OOM kill)"),
    Attack("file-descriptor-exhaustion", "open descriptors without bound", _DESCRIPTORS,
           _limited("opened", "RLIMIT_NOFILE", errors=("EMFILE",)), pod_controls="RLIMIT_NOFILE"),
    Attack("thread-exhaustion", "start threads without bound", _THREADS,
           lambda o: isinstance(o.report.get("started"), int) and o.report["started"] < o.rlimits["RLIMIT_NPROC"],
           pod_controls="threads count toward RLIMIT_NPROC and pids.max"),
    Attack("sleep-past-time-limit", "sleep past the wall-clock limit", _SLEEP,
           lambda o: o.timed_out and o.output is None and _no_survivors(o), timeout_seconds=3,
           pod_controls="watchdog wall-clock limit, then process sweep and Pod deletion"),
    Attack("spin-past-time-limit", "busy loop past the limit", _SPIN,
           lambda o: o.timed_out and o.output is None and _no_survivors(o), timeout_seconds=3,
           pod_controls="watchdog CPU fuses and wall-clock limit"),
    Attack("output-symlink-escape", "output.json as a symlink to a host file", _SYMLINK,
           _violation("output_symlink"), pod_controls="root collects output via O_PATH|O_NOFOLLOW"),
    Attack("output-proc-self-fd-escape", "output.json as a /proc/self/fd link", _PROC_FD,
           _violation("output_symlink"), pod_controls="same as output-symlink-escape"),
    Attack("output-hardlink-escape", "output.json hardlinked to another file", _HARDLINK,
           _violation("output_linked_or_foreign_file"), pod_controls="nlink == 1 and file uid check"),
    Attack("output-fifo-blocks-collector", "output.json as a FIFO to hang the supervisor", _FIFO,
           _violation("output_not_regular_file"), pod_controls="regular-file check before any open"),
    Attack("output-directory", "output.json as a directory", _DIRECTORY, _violation("output_not_regular_file")),
    Attack("output-oversized", "5 MiB output", _OVERSIZED, _violation("output_too_large"),
           pod_controls="4 MiB output bound below the 8 MiB control message"),
    Attack("output-malformed-json", "truncated JSON with NaN", _MALFORMED, _violation("output_not_json")),
    Attack("output-not-an-object", "JSON list instead of an object", _NOT_OBJECT, _violation("output_not_object")),
    Attack("forged-result-envelope", "fake supervisor envelope claiming success", _FORGED,
           lambda o: o.violation is None and o.submission_failed,
           pod_controls="supervisor builds the envelope; participant JSON only fills 'output'"),
    Attack("persistent-daemon", "setsid double-fork daemon and a file for the next run", _DAEMON,
           lambda o: _no_survivors(o) and o.report.get("daemon_started") is True,
           pod_controls="UID process sweep, then the one-shot Pod and its emptyDirs are deleted"),
    Attack("signals-and-setsid", "ignore SIGTERM/SIGHUP and detach", _SIGNALS,
           lambda o: _no_survivors(o) and o.report.get("daemon_started") is True,
           pod_controls="SIGKILL sweep; Pod deletion"),
    Attack("ptrace-supervisor", "PTRACE_ATTACH to the parent and PID 1", _PTRACE, _nothing_reached("reached"),
           pod_controls="different UID, no CAP_SYS_PTRACE, no_new_privs, gVisor ptrace emulation "
                        "(local run: same UID, blocked only by Yama scope 1)"),
    Attack("native-code-ctypes", "ctypes/libc; numpy and scipy must keep working; raise rlimits back",
           _NATIVE, lambda o: o.exit_code == 0 and o.report.get("reached") == []
           and all(o.report.get(k) is True for k in ("ctypes_ok", "numpy_ok", "scipy_ok")),
           pod_controls="hard rlimits cannot be raised without CAP_SYS_RESOURCE"),
    Attack("inherited-descriptors", "look for leaked supervisor descriptors", _INHERITED, _nothing_reached("reached"),
           pod_controls="subprocess close_fds; explicit child environment"),
    Attack("stdout-flood", "1 GiB of stdout", _STDOUT_FLOOD,
           lambda o: o.exit_code != 0 and o.stdout_bytes <= o.rlimits["RLIMIT_FSIZE"],
           pod_controls="RLIMIT_FSIZE on the log file; supervisor keeps 256 KiB"),
)


def attack(name: str) -> Attack:
    return next(item for item in ATTACKS if item.name == name)
