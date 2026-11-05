"""Typed state container for the VorLap GUI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from vorlap.structs import Component, VorLapAnalysisResult


StateListener = Callable[[str, object], None]


@dataclass
class VorLapGuiState:
    """Shared GUI state with lightweight change notifications."""

    components: list[Component] = field(default_factory=list)
    natural_frequencies: np.ndarray | None = None
    analysis_results: VorLapAnalysisResult | None = None
    qblade_sim_path: str = ""
    _listeners: list[StateListener] = field(default_factory=list, init=False, repr=False)

    def subscribe(self, listener: StateListener) -> StateListener:
        self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: StateListener) -> None:
        self._listeners = [current for current in self._listeners if current is not listener]

    def _notify(self, field_name: str, value: object) -> None:
        for listener in tuple(self._listeners):
            listener(field_name, value)

    def set_components(self, components: Sequence[Component]) -> None:
        self.components = list(components)
        if self.analysis_results is not None:
            self.set_analysis_results(None)
        self._notify("components", self.components)

    def set_natural_frequencies(self, natural_frequencies: np.ndarray | None) -> None:
        self.natural_frequencies = (
            None if natural_frequencies is None else np.asarray(natural_frequencies, dtype=float).copy()
        )
        self._notify("natural_frequencies", self.natural_frequencies)

    def set_analysis_results(self, analysis_results: VorLapAnalysisResult | None) -> None:
        self.analysis_results = analysis_results
        self._notify("analysis_results", self.analysis_results)

    def set_qblade_sim_path(self, qblade_sim_path: str | None) -> None:
        self.qblade_sim_path = str(qblade_sim_path or "")
        self._notify("qblade_sim_path", self.qblade_sim_path)
