from types import SimpleNamespace


def test_parent_scorer_waits_for_queued_children_without_restarting_the_score(monkeypatch):
    from mini_bang.framework import scoring_jobs
    clock=[0]
    states=iter(['pending','running','succeeded'])
    handle=SimpleNamespace(job_id='original-score')
    submitted=[]
    backend=SimpleNamespace(config=SimpleNamespace(job_timeout_seconds=10,polling_interval_seconds=1),
        submit=lambda job: submitted.append(job) or handle,
        poll=lambda current: SimpleNamespace(state=next(states)),fetch=lambda current:'saved score')
    monkeypatch.setattr(scoring_jobs.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(scoring_jobs.time,'sleep',lambda _:clock.__setitem__(0,clock[0]+10000))
    assert scoring_jobs.K8sScoringJobBackend.run(backend,'original input')=='saved score'
    assert submitted==['original input'] and clock[0]>10
