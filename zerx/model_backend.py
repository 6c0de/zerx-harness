"""The only module allowed to load/call the Gemma model. Defines a narrow
Protocol so every other module (and all local tests) can depend on
`ModelBackend` without ever importing a real model. `GemmaModelBackend`
loads the real thing and is exercised only on Colab/Kaggle — its
`generate()` is implemented in a later plan, never in local unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol


class ModelBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class FakeModelBackend:
    """Test double: returns scripted responses in order."""

    responses: List[str] = field(default_factory=list)
    _calls: List[str] = field(default_factory=list, init=False)

    def generate(self, prompt: str) -> str:
        self._calls.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeModelBackend: no scripted responses left")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_prompt(self) -> str:
        if not self._calls:
            raise RuntimeError("FakeModelBackend: generate() was never called")
        return self._calls[-1]


class GemmaModelBackend:
    """Real backend — loads Gemma-4-31B. Constructed but not exercised by
    local unit tests; Colab/Kaggle smoke tests cover this path per
    AGENTS.md's Colab and Kaggle gates.
    """

    def __init__(self, model_revision: str) -> None:
        self.model_revision = model_revision
        self._model = None  # loaded lazily by a later Colab/Kaggle-specific task

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "GemmaModelBackend.generate is implemented in the Colab/Kaggle "
            "model-loading plan, not the local model-free skeleton."
        )
