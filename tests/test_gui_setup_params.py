from __future__ import annotations

import numpy as np
import pytest

from vorlap.gui.tabs.setup import GUI_PARAMETER_SCHEMA, SimulationSetupTab


class ParamTable:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get_all(self):
        return [["", key, value] for key, value in self.values.items()]


class CapturingParamTable:
    def __init__(self) -> None:
        self.rows: list[list[str]] = []

    def append_row(self, row):
        self.rows.append(list(row))


def setup_tab_with_params(values: dict[str, str]) -> SimulationSetupTab:
    tab = SimulationSetupTab.__new__(SimulationSetupTab)
    tab.param_table = ParamTable(values)
    return tab


def test_default_parameter_rows_are_generated_from_schema_and_viv_profile():
    tab = SimulationSetupTab.__new__(SimulationSetupTab)
    tab.param_table = CapturingParamTable()

    tab._populate_default_params()

    row_by_param = {row[1]: row for row in tab.param_table.rows}
    defaults = SimulationSetupTab._default_param_values()
    assert list(row_by_param) == [name for name, _description in GUI_PARAMETER_SCHEMA]
    assert row_by_param["fluid_density"][2] == "1.225"
    assert row_by_param["azimuth_end"][2] == "360"
    assert row_by_param["inflow_speed_step"][2] == "4"
    assert row_by_param["output_time_end"][2] == "0.011"
    assert row_by_param["n_freq_depth"][2] == str(defaults["n_freq_depth"])


def test_get_viv_params_empty_table_uses_schema_defaults_without_tk():
    tab = setup_tab_with_params({})

    params = tab.get_viv_params()

    defaults = SimulationSetupTab._default_viv_params()
    np.testing.assert_allclose(params.azimuths, defaults.azimuths)
    np.testing.assert_allclose(params.inflow_speeds, defaults.inflow_speeds)
    np.testing.assert_allclose(params.output_time, defaults.output_time)
    assert params.n_harmonic == defaults.n_harmonic
    assert params.n_freq_depth == defaults.n_freq_depth
    assert params.amplitude_coeff_cutoff == pytest.approx(defaults.amplitude_coeff_cutoff)
    assert params.output_azimuth_vinf == defaults.output_azimuth_vinf


def test_get_viv_params_builds_nonempty_sweep_arrays_without_tk():
    tab = setup_tab_with_params(
        {
            "azimuth_start": "0",
            "azimuth_end": "20",
            "azimuth_step": "10",
            "inflow_speed_start": "2.0",
            "inflow_speed_end": "6.0",
            "inflow_speed_step": "2.0",
            "output_time_start": "0.0",
            "output_time_end": "0.003",
            "output_time_step": "0.001",
        }
    )

    params = tab.get_viv_params()

    np.testing.assert_allclose(params.azimuths, np.array([0.0, 10.0]))
    np.testing.assert_allclose(params.inflow_speeds, np.array([2.0, 4.0]))
    np.testing.assert_allclose(params.output_time, np.array([0.0, 0.001, 0.002]))


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"azimuth_step": "0"}, "Azimuth step must be positive"),
        ({"inflow_speed_step": "-1"}, "Inflow speed step must be positive"),
        (
            {"output_time_start": "0.1", "output_time_end": "0.0", "output_time_step": "0.01"},
            "Output time sweep is empty",
        ),
        ({"azimuth_start": "nan"}, "Azimuth start, end, and step must be finite values"),
    ],
)
def test_get_viv_params_rejects_invalid_sweep_ranges_without_compute(values, message):
    tab = setup_tab_with_params(values)

    with pytest.raises(ValueError, match=message):
        tab.get_viv_params()
