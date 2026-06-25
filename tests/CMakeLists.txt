add_executable(gary_tests test_core.cpp)
target_link_libraries(gary_tests PRIVATE gary_core)
add_test(NAME core COMMAND gary_tests)
