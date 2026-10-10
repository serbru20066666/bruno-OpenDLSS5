# Builds the shader and bench/vk/vk_gemm_bench.exe into build/bench (needs tools/: glslang, Vulkan-Headers, volk).
# If vcvars64.bat is not found automatically, set $env:VCVARS64 to its full path.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$vc = if ($env:VCVARS64) { $env:VCVARS64 } else { & (Join-Path $root "scripts\find_vcvars.ps1") }
$out = Join-Path $root "build\bench"
New-Item -ItemType Directory -Force $out | Out-Null
& (Join-Path $root "tools\glslang\bin\glslang.exe") -V --target-env vulkan1.3 (Join-Path $PSScriptRoot "gemm_f16_packed.comp") -o (Join-Path $out "gemm_f16_packed.spv")
$inc = '/I"' + (Join-Path $root "tools\Vulkan-Headers\include") + '" /I"' + (Join-Path $root "tools\volk") + '"'
$src = '"' + (Join-Path $PSScriptRoot "vk_gemm_bench.cpp") + '" "' + (Join-Path $root "tools\volk\volk.c") + '"'
Push-Location $out   # object files land here
cmd /c ('"' + $vc + '" >nul && cl /nologo /std:c++20 /EHsc /O2 /DNOMINMAX /DVK_ENABLE_BETA_EXTENSIONS ' + $inc + ' ' + $src + ' /Fe:vk_gemm_bench.exe')
Pop-Location
