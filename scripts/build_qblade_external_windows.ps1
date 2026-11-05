param(
    [string]$BuildDir = "build/qblade_external_windows",
    [string]$InstallDir = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$buildAbs = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $root $BuildDir }
if ($PythonExe -eq "") {
    $venvPy = Join-Path $root ".venv/Scripts/python.exe"
    if (Test-Path $venvPy) {
        $PythonExe = $venvPy
    } else {
        $PythonExe = "python"
    }
}

cmake -S (Join-Path $root "qblade_external_bridge") `
      -B $buildAbs `
      -DCMAKE_BUILD_TYPE=Release `
      -DPython3_EXECUTABLE=$PythonExe
cmake --build $buildAbs --config Release

$lib = Get-ChildItem -Path $buildAbs -Recurse -File | Where-Object {
    $_.Name -eq "libvorlap_qblade_bridge.dll"
} | Select-Object -First 1

if (-not $lib) {
    throw "Could not locate libvorlap_qblade_bridge.dll under $buildAbs"
}

Write-Host "Built library: $($lib.FullName)"

if ($InstallDir -ne "") {
    $installAbs = if ([System.IO.Path]::IsPathRooted($InstallDir)) { $InstallDir } else { Join-Path $root $InstallDir }
    New-Item -ItemType Directory -Path $installAbs -Force | Out-Null
    Copy-Item -Path $lib.FullName -Destination (Join-Path $installAbs $lib.Name) -Force
    Write-Host "Copied to: $installAbs"
}
