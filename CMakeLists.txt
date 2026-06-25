cmake_minimum_required(VERSION 3.20)
project(GARY VERSION 0.0.1 LANGUAGES CXX)

# ── Options ───────────────────────────────────────────────────────────────────
# CUDA is OFF by default: the core is portable C/C++ (CPU) and must build on a
# Raspberry Pi (ARM) with no CUDA present. Turn ON to build the Blackwell backend.
option(GARY_BUILD_CUDA  "Build the optional CUDA/Blackwell backend" OFF)
option(GARY_BUILD_TESTS "Build unit tests"                          OFF)

# ── Portable core: C++17, no CUDA dependency. x86 and ARM (Raspberry Pi). ──────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Portable core library (CPU, Pi-capable) ──────────────────────────────────
add_library(gary_core STATIC
  src/core/version.cpp
  src/core/info_theory.cpp
  src/core/signaling.cpp
)
target_include_directories(gary_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gary_core PUBLIC cxx_std_17)

# ── Optional CUDA / Blackwell backend ────────────────────────────────────────
if(GARY_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # sm_120 = Blackwell (RTX 50xx). Backend sources are added as the scale tiers land.
  set(CMAKE_CUDA_ARCHITECTURES 120)
  target_compile_definitions(gary_core PUBLIC GARY_WITH_CUDA=1)
  message(STATUS "GARY: CUDA backend ENABLED (sm_120 / Blackwell)")
else()
  message(STATUS "GARY: CUDA backend disabled — portable CPU core (Raspberry-capable)")
endif()

# ── Seed app (Phase 0 entry point) ───────────────────────────────────────────
add_executable(gary_seed apps/gary_seed.cpp)
target_link_libraries(gary_seed PRIVATE gary_core)

# ── Tests (optional) ─────────────────────────────────────────────────────────
if(GARY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
