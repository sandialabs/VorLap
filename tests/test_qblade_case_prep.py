from scripts.prepare_qblade_external_case import extract_structural_node_ids
from vorlap.qblade_runtime import (
    build_external_library_table_spec,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
)


def test_extract_structural_node_ids_filters_expected_targets():
    text = """
BLD_1_0.047438
BLD_1_0.047438
STR_1_2_0.172168
TWR_0.901408 // tower sensor
MOO_1_0.500000
"""
    ids = extract_structural_node_ids(text, include_tower_nodes=False)
    assert ids == ["BLD_1_0.047438", "STR_1_2_0.172168"]

    ids_with_tower = extract_structural_node_ids(text, include_tower_nodes=True)
    assert ids_with_tower == ["BLD_1_0.047438", "STR_1_2_0.172168", "TWR_0.901408"]


def test_structural_patch_is_idempotent_and_preserves_positions():
    base = """TWR_1.0\n"""
    spec = build_external_library_table_spec(["BLD_1_0.047438", "STR_2_3_0.174510"])

    first = patch_qblade_structural_definition(base, table_spec=spec)
    second = patch_qblade_structural_definition(first, table_spec=spec)

    assert first == second
    assert "EXTERNAL_1_IN" in second
    assert "EXTERNAL_1_OUT" in second
    assert '"X_g Vel. BLD_1 pos 0.047 [m/s]"' in second
    assert "ADDFORCE  BLD_1" in second
    assert "0.047438" in second
    assert "ADDFORCE  STR_2_3" in second
    assert "0.174510" in second


def test_turbine_patch_is_idempotent():
    text = """----------------------------------------Turbine Controller-----------------------------------------------------------
1                                                  CONTROLLERTYPE     - the type of turbine controller 0 = none, 1 = BLADED, 2 = DTU, 3 = TUB
libaero-lib-xflowenergy                            CONTROLLERFILE     - the controller file name, WITHOUT file ending (.dll or .so ) - leave blank if unused
Control/discon_xflowenergy.in                      PARAMETERFILE      - the controller parameter file name (leave blank if unused)
"""
    first = patch_qblade_turbine_definition(
        text,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=237,
        parameter_file="Control/vorlap_qblade_external.json",
    )
    second = patch_qblade_turbine_definition(
        first,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=237,
        parameter_file="Control/vorlap_qblade_external.json",
    )

    assert first == second
    assert "0                                                 CONTROLLERTYPE" in second
    assert "LIBARRAYSIZE_1" in second
    assert "237" in second
