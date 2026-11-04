"""Shared QBlade node-id parsing and channel naming helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Literal


QBladeNodeKind = Literal["blade", "strut", "tower"]

_POSITION_RE = r"[-+]?\d*\.?\d+"
_BLD_NODE_RE = re.compile(rf"^BLD_(?P<blade>\d+)_(?P<pos>{_POSITION_RE})$")
_STR_NODE_RE = re.compile(rf"^STR_(?P<strut>\d+)_(?P<blade>\d+)_(?P<pos>{_POSITION_RE})$")
_TWR_NODE_RE = re.compile(rf"^TWR_(?P<section>\d+)_(?P<pos>{_POSITION_RE})$")
_TWR_SHORT_RE = re.compile(rf"^TWR_(?P<pos>{_POSITION_RE})$")


@dataclass(frozen=True)
class QBladeNodeTarget:
    """Parsed QBlade aerodynamic/structural node target."""

    node_id: str
    kind: QBladeNodeKind
    action_id: str
    position: float

    @property
    def component_id(self) -> str:
        """VorLap/QBlade component id associated with this node."""
        return self.action_id

    @property
    def canonical_node_id(self) -> str:
        """Canonical node id with the normalized position rendered to six decimals."""
        return canonical_node_id(self.action_id, self.position)


def _coerce_position(raw_position: str, node_id: str, *, validate_position: bool) -> float:
    position = float(raw_position)
    if not math.isfinite(position):
        raise ValueError(f"QBlade node id '{node_id}' has non-finite normalized position {position!r}.")
    if validate_position and (position < 0.0 or position > 1.0):
        raise ValueError(
            f"QBlade node id '{node_id}' has normalized position {position:g}; "
            "positions must be finite and within [0, 1]."
        )
    return position


def parse_qblade_node_id(node_id: Any, *, validate_position: bool = True) -> QBladeNodeTarget:
    """Parse a QBlade node id into an action/component target and normalized position."""
    node = str(node_id).strip()

    match = _BLD_NODE_RE.match(node)
    if match:
        return QBladeNodeTarget(
            node_id=node,
            kind="blade",
            action_id=f"BLD_{match.group('blade')}",
            position=_coerce_position(match.group("pos"), node, validate_position=validate_position),
        )

    match = _STR_NODE_RE.match(node)
    if match:
        return QBladeNodeTarget(
            node_id=node,
            kind="strut",
            action_id=f"STR_{match.group('strut')}_{match.group('blade')}",
            position=_coerce_position(match.group("pos"), node, validate_position=validate_position),
        )

    match = _TWR_NODE_RE.match(node)
    if match:
        return QBladeNodeTarget(
            node_id=node,
            kind="tower",
            action_id=f"TWR_{match.group('section')}",
            position=_coerce_position(match.group("pos"), node, validate_position=validate_position),
        )

    match = _TWR_SHORT_RE.match(node)
    if match:
        return QBladeNodeTarget(
            node_id=node,
            kind="tower",
            action_id="TWR_1",
            position=_coerce_position(match.group("pos"), node, validate_position=validate_position),
        )

    raise ValueError(f"Unsupported QBlade node id format: '{node_id}'.")


def is_qblade_node_id(node_id: Any, *, include_tower: bool = True, validate_position: bool = False) -> bool:
    """Return true when text is a supported QBlade blade, strut, or tower node id."""
    try:
        parsed = parse_qblade_node_id(node_id, validate_position=validate_position)
    except (TypeError, ValueError):
        return False
    return include_tower or parsed.kind != "tower"


def action_target_from_node_id(node_id: Any, *, validate_position: bool = True) -> tuple[str, float]:
    """Return QBlade external-action target `(action_id, position)` for a node id."""
    parsed = parse_qblade_node_id(node_id, validate_position=validate_position)
    return parsed.action_id, parsed.position


def component_id_from_node_id(node_id: Any, *, validate_position: bool = True) -> str:
    """Return the QBlade/VorLap component id for a node id."""
    return parse_qblade_node_id(node_id, validate_position=validate_position).component_id


def canonical_node_id(component_id: Any, position: float) -> str:
    """Render a component and normalized position as the canonical node id."""
    return f"{component_id}_{float(position):.6f}"


def action_position_label(action_id: Any, position: float) -> str:
    """Render the human-readable QBlade variable suffix for an action target."""
    action = str(action_id)
    pos = float(position)
    if action.startswith(("BLD_", "STR_")):
        return f"{action} pos {pos:.3f}"
    if action.startswith("TWR"):
        return f"TWR pos {pos:.3f}"
    if action.startswith("TRQ"):
        return f"TRQ pos {pos:.3f}"
    return f"{action} pos {pos:.3f}"


def node_position_label(node_id: Any, *, validate_position: bool = True) -> str:
    """Render the human-readable QBlade variable suffix for a node id."""
    action_id, position = action_target_from_node_id(node_id, validate_position=validate_position)
    return action_position_label(action_id, position)


def velocity_channel_name(node_id: Any, axis: str, *, validate_position: bool = True) -> str:
    """Return the QBlade global structural-velocity channel name for a node id and axis."""
    return f"{str(axis).upper()}_g Vel. {node_position_label(node_id, validate_position=validate_position)} [m/s]"


def position_channel_name(action: dict[str, Any]) -> str:
    """Return the QBlade global position-channel name for a constant-force action."""
    axis = str(action["direction"]).upper()
    label = action_position_label(str(action["id"]), float(action["position"]))
    return f"{axis}_g Pos. {label} [m]"


def extract_structural_node_ids(
    structural_text: str,
    *,
    include_tower_nodes: bool = False,
    unique: bool = True,
) -> list[str]:
    """Extract blade/strut and optional tower node ids from QBlade structural text."""
    node_ids: list[str] = []
    seen: set[str] = set()
    for raw in str(structural_text).splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if not is_qblade_node_id(line, include_tower=include_tower_nodes, validate_position=False):
            continue
        if unique and line in seen:
            continue
        if unique:
            seen.add(line)
        node_ids.append(line)
    return node_ids


def infer_geometry_inclusions(
    node_ids: Iterable[Any],
    include_struts: bool,
    include_tower: bool,
) -> tuple[bool, bool]:
    """Force strut/tower conversion when requested node ids require those geometry classes."""
    requires_struts = False
    requires_tower = False
    for node_id in node_ids:
        try:
            parsed = parse_qblade_node_id(node_id, validate_position=False)
        except ValueError:
            text = str(node_id)
            requires_struts = requires_struts or text.startswith("STR_")
            requires_tower = requires_tower or text.startswith("TWR_")
            continue
        requires_struts = requires_struts or parsed.kind == "strut"
        requires_tower = requires_tower or parsed.kind == "tower"
    return bool(include_struts or requires_struts), bool(include_tower or requires_tower)


__all__ = [
    "QBladeNodeTarget",
    "action_position_label",
    "action_target_from_node_id",
    "canonical_node_id",
    "component_id_from_node_id",
    "extract_structural_node_ids",
    "infer_geometry_inclusions",
    "is_qblade_node_id",
    "node_position_label",
    "parse_qblade_node_id",
    "position_channel_name",
    "velocity_channel_name",
]
