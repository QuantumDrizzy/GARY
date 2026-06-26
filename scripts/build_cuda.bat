@echo off
REM GARY - build the optional CUDA / Blackwell backend on Windows.
REM Needs: VS 2022 (Build Tools or IDE), the CUDA Toolkit, CMake, and Ninja.
REM The VS CMake generator does not pick up the CUDA toolset here, so we use Ninja.
REM No hardcoded user paths: nvcc/ninja are resolved from PATH or standard install dirs.
REM Override the MSVC env script via the VCVARS env var if your install differs.

if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" || (echo [GARY] Set the VCVARS env var to your vcvars64.bat path. & exit /b 1)

REM Put nvcc / ninja on PATH if they are not already (CUDA_PATH + winget dir are user-agnostic).
where nvcc  >nul 2>nul || set "PATH=%CUDA_PATH%\bin;%PATH%"
where ninja >nul 2>nul || set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"

cd /d "%~dp0.."
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGARY_BUILD_CUDA=ON -DGARY_BUILD_TESTS=OFF || exit /b 1
cmake --build build-cuda
