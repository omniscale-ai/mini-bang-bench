"""Explicit judge authority for trusted scoring only, never public task contexts."""
import os


PRIVATE_JUDGE_ENVIRONMENT = (
    "MINI_BANG_JUDGE_URL", "MINI_BANG_JUDGE_MODEL", "MINI_BANG_JUDGE_TOKEN",
    "MINI_BANG_EVALUATED_MODEL", "MINI_BANG_JUDGE_VERSION", "MINI_BANG_JUDGE_PROFILE",
    "MINI_BANG_JUDGE_SAME_MODEL_AUTHORIZATION", "MINI_BANG_JUDGE_HTTP_TIMEOUT",
    "MINI_BANG_REFERENCE_BUNDLE_VERSION", "MINI_BANG_REVIEWED_SOURCE_PROFILE",
)
# Cluster settings that hidden scoring needs; simulators add their own `runtime_environment`.
SCORING_INFRASTRUCTURE_ENVIRONMENT = (
    "MINI_BANG_K8S_STAGER_IMAGE", "MINI_BANG_K8S_STAGING_STORAGE_CLASS", "MINI_BANG_K8S_GPU_RUNTIME_CLASS",
    "MINI_BANG_K8S_GPU_WORKER_IMAGE", "MINI_BANG_K8S_GPU_NODE_SELECTOR",
)


def private_judge_environment():
    return {name: os.environ[name] for name in PRIVATE_JUDGE_ENVIRONMENT if name in os.environ}


def scoring_deployment_environment():
    """Simulator runtime settings, then cluster settings, as set in the runner environment."""
    from mini_bang.simulators.registry import runtime_environment_names
    names = (*runtime_environment_names(), *SCORING_INFRASTRUCTURE_ENVIRONMENT)
    return {name: os.environ[name] for name in names if name in os.environ}
