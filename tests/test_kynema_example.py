"""Exercise preparation only; no solver, mesh license or Slurm submission."""
from pathlib import Path
import subprocess
import sys

import yaml

SCRIPT = Path(__file__).resolve().parents[1]/'examples/naca0018_kynema/nalu_runs/submission_scripts/gen_submit_nalu.py'


def test_prepare_is_additive_and_does_not_submit(tmp_path):
    # An existing stand-in tests path/YAML plumbing, not Exodus validity.
    mesh = tmp_path/'mesh.exo'
    mesh.write_bytes(b'path fixture, not a real CFD mesh')
    output = tmp_path/'cases'
    command = [sys.executable,str(SCRIPT),'--mesh',str(mesh),'--output',str(output),
               '--aoa','84','--speed','3.75']
    first = subprocess.run(command,capture_output=True,text=True,check=True)
    assert 'DRY RUN:' in first.stdout
    assert 'Re=500000' in first.stdout
    target = next(output.rglob('*.yaml'))
    before = target.read_bytes()
    data = yaml.safe_load(before)
    integrator = data['Time_Integrators'][0]['StandardTimeIntegrator']
    assert integrator['time_step'] > 0
    assert integrator['termination_step_count'] > 100
    assert Path(data['realms'][0]['mesh']) == mesh
    assert all(Path(s['hypre_cfg_file']).is_file() for s in data['linear_solvers'])
    second = subprocess.run(command,capture_output=True,text=True)
    assert second.returncode != 0
    assert 'FileExistsError' in second.stderr
    assert target.read_bytes() == before
