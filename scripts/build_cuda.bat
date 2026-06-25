@echo off
REM GARY — build the optional CUDA/Blackwell backend on Windows.
REM Loads the MSVC environment, puts nvcc + ninja on PATH, configures with Ninja
REM (the VS generator does not pick up the CUDA toolset here), and builds Release.
REM Paths below match this machine (VS BuildTools 2022 + CUDA v13.0 + winget Ninja);
REM adjust if yours differ.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;C:\Users\Drizzy\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
cd /d "%~dp0.."

cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGARY_BUILD_CUDA=ON -DGARY_BUILD_TESTS=OFF
if errorlevel 1 exit /b 1
cmake --build build-cuda
