"""Failures proven by a canonical worker result, distinct from harness errors."""


class GenuineObservationFailure(RuntimeError):
    """The worker rejected admitted work; private details never become UI text."""

    def __init__(self, detail, handle):
        super().__init__('Observation request was rejected by the worker')
        self.private_detail = detail
        self.original_handle = handle
