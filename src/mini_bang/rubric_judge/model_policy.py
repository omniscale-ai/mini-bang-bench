"""Model authority for the reviewed description judge protocol."""
REVIEWED_JUDGE_VERSION = "reviewed-distributed-spatial"
REVIEWED_JUDGE_PROFILE = "distributed-spatial"
REVIEWED_JUDGE_MODEL = "judge-model"
SAME_MODEL_AUTHORIZATION = "same-model-judge-authorized"
AUTHORIZATION_ENV = "MINI_BANG_JUDGE_SAME_MODEL_AUTHORIZATION"


def model_authority(*, version, profile, model, evaluated_model, authorization=None):
    """Judge and evaluated model may be equal only for the exact reviewed protocol with explicit authorization."""
    if version == REVIEWED_JUDGE_VERSION and model != REVIEWED_JUDGE_MODEL:
        raise ValueError("This judge protocol requires exactly its reviewed judge model")
    if version == REVIEWED_JUDGE_VERSION and profile != REVIEWED_JUDGE_PROFILE:
        raise ValueError("This judge protocol requires its exact reviewed judge profile")
    equal = model == evaluated_model
    if equal and not (version == REVIEWED_JUDGE_VERSION and profile == REVIEWED_JUDGE_PROFILE
            and model == REVIEWED_JUDGE_MODEL and authorization == SAME_MODEL_AUTHORIZATION):
        raise ValueError("Same-model judging lacks the exact explicit authorization")
    return {"mode": "same_model_explicit_user_request" if equal else "different_models",
        "authorization": SAME_MODEL_AUTHORIZATION if equal else None,
        "judge_model": model, "evaluated_model": evaluated_model,
        "judge_version": version, "judge_profile": profile}
