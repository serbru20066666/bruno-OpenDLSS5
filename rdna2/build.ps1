# Builds the RDNA2 shaders and rdna2_net.exe into build/rdna2 (needs tools/: glslang, Vulkan-Headers, volk).
# If vcvars64.bat is not found automatically, set $env:VCVARS64 to its full path.
$root = Split-Path -Parent $PSScriptRoot
$vc = if ($env:VCVARS64) { $env:VCVARS64 } else { & (Join-Path $root "scripts\find_vcvars.ps1") }
$out = Join-Path $root "build\rdna2"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($n in "gemm", "wattn", "gattn", "ops") {
  & (Join-Path $root "tools\glslang\bin\glslang.exe") -V --target-env vulkan1.3 (Join-Path $PSScriptRoot "shaders\$n.comp") -o (Join-Path $out "$n.spv")
  if ($LASTEXITCODE -ne 0) { throw "shader $n failed" }
}
$inc = '/I"' + (Join-Path $root "tools\Vulkan-Headers\include") + '" /I"' + (Join-Path $root "tools\volk") + '"'
$src = '"' + (Join-Path $PSScriptRoot "rdna2_net.cpp") + '" "' + (Join-Path $root "tools\volk\volk.c") + '"'
Push-Location $out   # object files land here
cmd /c ('"' + $vc + '" >nul && cl /nologo /std:c++20 /EHsc /O2 /DNOMINMAX /DVK_ENABLE_BETA_EXTENSIONS ' + $inc + ' ' + $src + ' /Fe:rdna2_net.exe')
Pop-Location
