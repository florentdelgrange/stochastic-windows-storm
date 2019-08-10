# Base path for test files
set(STORM_TESTS_BASE_PATH "${PROJECT_SOURCE_DIR}/src/test/stochastic-windows")

# Test Sources
file(GLOB_RECURSE ALL_FILES ${STORM_TESTS_BASE_PATH}/*.h ${STORM_TESTS_BASE_PATH}/*.cpp)

register_source_groups_from_filestructure("${ALL_FILES}" test)

# Note that the tests also need the source files, except for the main file
include_directories(${GTEST_INCLUDE_DIR})

foreach (testsuite objectives)

	  file(GLOB_RECURSE TEST_${testsuite}_FILES ${STORM_TESTS_BASE_PATH}/${testsuite}/*.h ${STORM_TESTS_BASE_PATH}/${testsuite}/*.cpp)
      add_executable (test-sw-${testsuite} ${TEST_${testsuite}_FILES} ${STORM_TESTS_BASE_PATH}/storm-test.cpp)
	  target_link_libraries(test-sw-${testsuite} stochastic-windows)
	  target_link_libraries(test-sw-${testsuite} ${STORM_TEST_LINK_LIBRARIES})

	  add_dependencies(test-sw-${testsuite} test-resources)
	  add_test(NAME run-test-sw-${testsuite} COMMAND $<TARGET_FILE:test-sw-${testsuite}>)
	  add_dependencies(tests test-sw-${testsuite})

endforeach ()