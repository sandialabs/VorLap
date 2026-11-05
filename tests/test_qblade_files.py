from pathlib import Path

import pytest

from vorlap.qblade_files import (
    first_qblade_keyword_value,
    format_qblade_keyword_line,
    is_absolute_qblade_path,
    parse_qblade_keyword_line,
    parse_qblade_keywords,
    read_qblade_float_keyword,
    read_qblade_keyword,
    replace_qblade_keyword_file,
    replace_qblade_keyword_text,
    resolve_qblade_path,
    resolve_qblade_path_text,
)
from vorlap.qblade_parser import load_qblade_blade_definition, load_qblade_turbine_definition


def test_parse_qblade_keyword_line_accepts_value_leading_comments_and_spaced_values():
    assert parse_qblade_keyword_line("case.trb TURBFILE - turbine file") == ("TURBFILE", "case.trb")
    assert parse_qblade_keyword_line("Control/my config.json LIBPARAMETERFILE_1 - config") == (
        "LIBPARAMETERFILE_1",
        "Control/my config.json",
    )
    assert parse_qblade_keyword_line("----------------------------------------Header---") is None
    assert parse_qblade_keyword_line("TURB_1") is None


def test_parse_qblade_keyword_line_preserves_negative_values():
    assert parse_qblade_keyword_line("-50.0 VERTANGLE - vertical inflow angle") == (
        "VERTANGLE",
        "-50.0",
    )
    assert parse_qblade_keyword_line("----------------------------------------") is None


def test_parse_qblade_keywords_preserves_repeated_values(tmp_path: Path):
    qblade_file = tmp_path / "case.sim"
    qblade_file.write_text(
        "first.trb TURBFILE - first\n"
        "second.trb TURBFILE - second\n"
        "0.0075 TIMESTEP - timestep\n",
        encoding="utf-8",
    )

    parsed = parse_qblade_keywords(qblade_file)

    assert parsed["TURBFILE"] == ["first.trb", "second.trb"]
    assert first_qblade_keyword_value(parsed, "TURBFILE") == "first.trb"
    assert read_qblade_keyword(qblade_file, "TIMESTEP") == "0.0075"
    assert read_qblade_float_keyword(qblade_file, "TIMESTEP") == pytest.approx(0.0075)


def test_resolve_qblade_path_uses_existing_fallback_bases(tmp_path: Path):
    config = tmp_path / "case" / "Control" / "vorlap_qblade_external.json"
    config.parent.mkdir(parents=True)
    fallback = tmp_path / "source" / "airfoils"
    fallback.mkdir(parents=True)

    resolved = resolve_qblade_path(config, "airfoils", extra_bases=[tmp_path / "source"])

    assert resolved == fallback.resolve()
    assert resolve_qblade_path_text(config, "airfoils", extra_bases=[tmp_path / "source"]) == str(
        fallback.resolve()
    )


def test_resolve_qblade_path_keeps_absolute_path_text(tmp_path: Path):
    absolute = tmp_path / "wind.wnd"
    assert resolve_qblade_path(tmp_path / "case.sim", absolute) == absolute.resolve()
    assert is_absolute_qblade_path(r"C:\QBlade\case.sim")
    assert resolve_qblade_path_text(tmp_path / "case.sim", r"C:\QBlade\case.sim") == r"C:\QBlade\case.sim"


def test_format_and_replace_qblade_keyword_text():
    assert (
        format_qblade_keyword_line("0.0075", "TIMESTEP", "the timestep")
        == "0.0075                                            TIMESTEP           - the timestep"
    )

    text = "0.01 TIMESTEP - old comment\n"
    patched, old_value, count = replace_qblade_keyword_text(text, "TIMESTEP", 0.0075, comment=None)

    assert count == 1
    assert old_value == "0.01"
    assert patched == f"{'0.0075':<50}{'TIMESTEP':<18} - old comment\n"

    patched_no_comment, _old_value, count = replace_qblade_keyword_text(
        text,
        "TIMESTEP",
        0.0075,
        comment="new comment",
    )

    assert count == 1
    assert patched_no_comment == "0.0075                                            TIMESTEP           - new comment\n"


def test_replace_qblade_keyword_file_returns_patch_metadata(tmp_path: Path):
    qblade_file = tmp_path / "case.sim"
    qblade_file.write_text("2000 NUMTIMESTEPS - steps\n", encoding="utf-8")

    metadata = replace_qblade_keyword_file(qblade_file, "NUMTIMESTEPS", 20, comment="")

    assert metadata == {
        "file": str(qblade_file),
        "keyword": "NUMTIMESTEPS",
        "old_value": "2000",
        "new_value": "20",
    }
    assert qblade_file.read_text(encoding="utf-8") == f"{'20':<50}{'NUMTIMESTEPS':<18}\n"


def test_replace_empty_value_keyword_line_does_not_consume_previous_line():
    text = (
        "0 WNDTYPE - wind type\n"
        " WNDNAME - wind file\n"
        "0 RELAXSTEPS - static relaxation\n"
    )

    patched, old_value, count = replace_qblade_keyword_text(text, "WNDNAME", "wind.bts", comment=None)

    assert count == 1
    assert old_value == ""
    assert "0 WNDTYPE - wind type\n" in patched
    assert f"{'wind.bts':<50}{'WNDNAME':<18} - wind file\n" in patched
    assert "0 RELAXSTEPS - static relaxation\n" in patched


def test_replace_qblade_keyword_keeps_separator_after_long_absolute_value(tmp_path: Path):
    wind_name = (
        "/pscratch/kevmoor/vorlap_wind_fields/changed_geometry_trial/"
        "U05p00ms/dlc6p4_U05p00ms.bts"
    )
    qblade_file = tmp_path / "case.sim"
    qblade_file.write_text(" WNDNAME - wind file\n", encoding="utf-8")

    replace_qblade_keyword_file(qblade_file, "WNDNAME", wind_name, comment="wind file")

    line = qblade_file.read_text(encoding="utf-8")
    assert f"{wind_name} WNDNAME" in line
    assert read_qblade_keyword(qblade_file, "WNDNAME") == wind_name


def test_qblade_parsers_recognize_hawt_type_and_station_table(tmp_path: Path):
    trb = tmp_path / "blade.trb"
    trb.write_text("0 TURBTYPE - HAWT\n1 NUMBLADES - blade count\n", encoding="utf-8")
    bld = tmp_path / "blade.bld"
    bld.write_text(
        "HAWT ROTORTYPE - rotor type\n"
        "1 NUMBLADES - blade count\n"
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE\n"
        "0.0 4.0 10.0 0.1 -0.2 0.5 Polars/FFA-W3-360.plr\n"
        "90.0 1.0 0.0 0.3 -1.2 0.45 Polars/FFA-W3-211.plr\n",
        encoding="utf-8",
    )

    assert load_qblade_turbine_definition(str(trb))["turbine_type"] == 0
    parsed = load_qblade_blade_definition(str(bld))
    assert parsed["rotor_type"] == "HAWT"
    assert len(parsed["blade_rows"]) == 2
    assert parsed["blade_rows"][1]["POS_[m]"] == "90.0"
