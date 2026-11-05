import pytest

from vorlap.qblade_nodes import (
    action_position_label,
    action_target_from_node_id,
    component_id_from_node_id,
    extract_structural_node_ids,
    infer_geometry_inclusions,
    parse_qblade_node_id,
    position_channel_name,
    velocity_channel_name,
)


@pytest.mark.parametrize(
    ("node_id", "kind", "action_id", "position", "canonical"),
    [
        ("BLD_2_0.230455", "blade", "BLD_2", 0.230455, "BLD_2_0.230455"),
        ("STR_1_3_0.172168", "strut", "STR_1_3", 0.172168, "STR_1_3_0.172168"),
        ("TWR_4_0.901408", "tower", "TWR_4", 0.901408, "TWR_4_0.901408"),
        ("TWR_0.901408", "tower", "TWR_1", 0.901408, "TWR_1_0.901408"),
    ],
)
def test_parse_qblade_node_id_returns_action_target(node_id, kind, action_id, position, canonical):
    parsed = parse_qblade_node_id(node_id)

    assert parsed.node_id == node_id
    assert parsed.kind == kind
    assert parsed.action_id == action_id
    assert parsed.component_id == action_id
    assert parsed.position == pytest.approx(position)
    assert parsed.canonical_node_id == canonical
    target_action_id, target_position = action_target_from_node_id(node_id)
    assert target_action_id == action_id
    assert target_position == pytest.approx(position)
    assert component_id_from_node_id(node_id) == action_id


@pytest.mark.parametrize("node_id", ["BLD_1_1.500000", "STR_1_2_-0.100000", "TWR_1_1.250000"])
def test_parse_qblade_node_id_rejects_out_of_range_positions(node_id):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        parse_qblade_node_id(node_id)

    parsed = parse_qblade_node_id(node_id, validate_position=False)
    assert isinstance(parsed.position, float)


def test_parse_qblade_node_id_rejects_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported QBlade node id format"):
        parse_qblade_node_id("MOO_1_0.500000")


def test_extract_structural_node_ids_filters_expected_targets_and_preserves_order():
    text = """
BLD_1_0.047438
BLD_1_0.047438
STR_1_2_0.172168
TWR_0.901408 // tower sensor
MOO_1_0.500000
"""
    assert extract_structural_node_ids(text, include_tower_nodes=False) == [
        "BLD_1_0.047438",
        "STR_1_2_0.172168",
    ]
    assert extract_structural_node_ids(text, include_tower_nodes=True) == [
        "BLD_1_0.047438",
        "STR_1_2_0.172168",
        "TWR_0.901408",
    ]
    assert extract_structural_node_ids(text, include_tower_nodes=True, unique=False) == [
        "BLD_1_0.047438",
        "BLD_1_0.047438",
        "STR_1_2_0.172168",
        "TWR_0.901408",
    ]


def test_infer_geometry_inclusions_from_node_ids():
    include_struts, include_tower = infer_geometry_inclusions(
        ["BLD_1_0.047438", "STR_1_2_0.172168", "TWR_0.901408"],
        include_struts=False,
        include_tower=False,
    )

    assert include_struts is True
    assert include_tower is True


def test_qblade_channel_labels_match_bridge_conventions():
    assert velocity_channel_name("BLD_1_0.047438", "x") == "X_g Vel. BLD_1 pos 0.047 [m/s]"
    assert action_position_label("TWR_1", 0.901408) == "TWR pos 0.901"
    assert (
        position_channel_name(
            {
                "direction": "Y",
                "id": "STR_1_3",
                "position": 0.172168,
            }
        )
        == "Y_g Pos. STR_1_3 pos 0.172 [m]"
    )
