import os
from pathlib import Path
import subprocess
import sys

from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.client.deployment.public_agent_sdk import public_agent_sdk, sdk_volume
from mini_bang.framework.answer_format import example_answer
from mini_bang.framework.runtime import RunManager
from tests.test_official_boundaries import _server


def test_public_sdk_runs_an_agent_with_stdlib_only_and_no_benchmark_internals(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_BANG_OBSERVATION_RECEIPTS_DIR", str(tmp_path / "state"))
    sdk, volume = public_agent_sdk("test"), sdk_volume()
    sdk_root = tmp_path / "public-sdk"
    for item in volume["configMap"]["items"]:
        target = sdk_root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sdk["data"][item["key"]])
    server = OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
                              mode="multi_session_official", admin_token="organizer")
    session = server.create_session(task_id="raf/mechanism-prob")
    answer = example_answer(session.task_spec, session.instance_spec)
    agent = tmp_path / "example_agent.py"
    agent.write_text(f'''import importlib.util
from mini_bang.framework import TaskSubmission
from mini_bang.framework.task import TaskEnvironment
assert importlib.util.find_spec('mini_bang.tasks') is None
assert importlib.util.find_spec('mini_bang.backends') is None
assert importlib.util.find_spec('mini_bang.pool') is None
class ExampleAgent:
 def solve(self, task):
  assert isinstance(task,TaskEnvironment)
  assert task.api.instructions()
  assert task.metadata['task_id'] == 'raf/mechanism-prob'
  return TaskSubmission(answer={answer!r})
''')
    try:
        with _server(server) as url:
            env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(sdk_root),
                "MINI_BANG_RUNNER_URL": url, "MINI_BANG_RUN_ID": session.run_id,
                "MINI_BANG_RUNNER_TOKEN": server.session_token(session.run_id)}
            # -S excludes site-packages/editable checkouts, as in a bare Python image.
            result = subprocess.run([sys.executable, "-S", "-m", "mini_bang.agent_runtime",
                "--agent-file", str(agent) + ":ExampleAgent"], env=env, cwd=tmp_path,
                text=True, capture_output=True, timeout=15)
            assert result.returncode == 0, result.stderr
            assert server._submissions[session.run_id].answer == answer
    finally:
        server.observations.close()


def test_agent_client_directory_has_only_the_client_and_runs_without_site_packages(tmp_path, capsys):
    from mini_bang.cli import main
    from mini_bang.client.deployment.public_agent_sdk import public_agent_sdk

    output = tmp_path / "agent-client"
    assert main(["agent-client", "--output", str(output), "--format", "json"]) == 0
    assert '"files"' in capsys.readouterr().out
    written = sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
    assert written == ["mini_bang/__init__.py", "mini_bang/agent_runtime.py", "mini_bang/agent_types.py",
                       "mini_bang/framework/__init__.py", "mini_bang/framework/task.py", "mini_bang/importing.py",
                       "probe.py", "pyproject.toml"]
    assert "def validate_probe_leave_pair_out(" in (output / "probe.py").read_text()
    assert 'dependencies = []' in (output / "pyproject.toml").read_text()
    mounted = public_agent_sdk("test")["data"]
    assert (output / "mini_bang/agent_runtime.py").read_text() == mounted["agent_runtime.py"]
    probe = ("import importlib.util as u, mini_bang.agent_runtime, mini_bang.framework;"
             "assert all(u.find_spec(m) is None for m in "
             "('mini_bang.simulators', 'mini_bang.tasks', 'mini_bang.backends', 'numpy'))")
    result = subprocess.run([sys.executable, "-S", "-c", probe], env={"PATH": os.environ["PATH"],
                            "PYTHONPATH": str(output)}, cwd=tmp_path, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
