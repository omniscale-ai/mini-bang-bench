FROM python:3.12-slim

WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src
ENV MINI_BANG_AGENT_FILE=/workspace/examples/official_timing_agent.py:OfficialTimingAgent
CMD ["python", "-m", "mini_bang.agent_runtime"]
