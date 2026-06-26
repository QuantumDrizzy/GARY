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
  src/core/population_cpu.cpp
  src/core/dark_forest.cpp
  src/core/decipher.cpp
  src/core/quantum.cpp
  src/core/chaos.cpp
  src/core/iterated.cpp
  src/core/bifurcation.cpp
  src/core/quantum_learning.cpp
  src/core/henon.cpp
  src/core/neural.cpp
  src/core/neural_compose.cpp
)
target_include_directories(gary_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gary_core PUBLIC cxx_std_17)

# ── Optional CUDA / Blackwell backend ────────────────────────────────────────
if(GARY_BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # sm_120 = Blackwell (RTX 50xx). Set on the target (it already exists above).
  target_sources(gary_core PRIVATE src/cuda/population_cuda.cu)
  set_target_properties(gary_core PROPERTIES CUDA_ARCHITECTURES 120)
  target_compile_definitions(gary_core PUBLIC GARY_WITH_CUDA=1)
  target_link_libraries(gary_core PUBLIC CUDA::cudart CUDA::curand)
  message(STATUS "GARY: CUDA backend ENABLED (sm_120 / Blackwell)")
else()
  message(STATUS "GARY: CUDA backend disabled — portable CPU core (Raspberry-capable)")
endif()

# ── Seed app (Phase 0 entry point) ───────────────────────────────────────────
add_executable(gary_seed apps/gary_seed.cpp)
target_link_libraries(gary_seed PRIVATE gary_core)

add_executable(gary_population apps/gary_population.cpp)
target_link_libraries(gary_population PRIVATE gary_core)

add_executable(gary_darkforest apps/gary_darkforest.cpp)
target_link_libraries(gary_darkforest PRIVATE gary_core)

add_executable(gary_lingua apps/gary_lingua.cpp)
target_link_libraries(gary_lingua PRIVATE gary_core)

add_executable(gary_quantum apps/gary_quantum.cpp)
target_link_libraries(gary_quantum PRIVATE gary_core)

add_executable(gary_chaos apps/gary_chaos.cpp)
target_link_libraries(gary_chaos PRIVATE gary_core)

add_executable(gary_compose apps/gary_compose.cpp)
target_link_libraries(gary_compose PRIVATE gary_core)

add_executable(gary_bifurcation apps/gary_bifurcation.cpp)
target_link_libraries(gary_bifurcation PRIVATE gary_core)

add_executable(gary_qlearn apps/gary_qlearn.cpp)
target_link_libraries(gary_qlearn PRIVATE gary_core)

add_executable(gary_henon apps/gary_henon.cpp)
target_link_libraries(gary_henon PRIVATE gary_core)

add_executable(gary_neural apps/gary_neural.cpp)
target_link_libraries(gary_neural PRIVATE gary_core)

add_executable(gary_ncompose apps/gary_ncompose.cpp)
target_link_libraries(gary_ncompose PRIVATE gary_core)

add_executable(gary_disentangle apps/gary_disentangle.cpp)
target_link_libraries(gary_disentangle PRIVATE gary_core)

add_executable(gary_niterate apps/gary_niterate.cpp)
target_link_libraries(gary_niterate PRIVATE gary_core)

# ── Tests (optional) ─────────────────────────────────────────────────────────
if(GARY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
