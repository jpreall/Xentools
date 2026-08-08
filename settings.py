"""Package-wide runtime settings for xentools."""

from __future__ import annotations


class XentoolsSettings:
    """
    Runtime settings controlling user-facing package behavior.

    ``verbosity`` is intentionally simple for now:

    - ``0``: suppress informational xentools warnings.
    - ``1``: default; show helpful warnings about expensive operations.
    """

    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: int):
        value = int(value)
        if value < 0:
            raise ValueError("verbosity must be >= 0.")
        self._verbosity = value


settings = XentoolsSettings()
