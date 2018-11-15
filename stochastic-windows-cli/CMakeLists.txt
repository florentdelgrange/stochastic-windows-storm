# Create stochastic windows.

file(GLOB_RECURSE STOCHASTIC_WINDOWS_CLI_SOURCES ${PROJECT_SOURCE_DIR}/src/stochastic-windows-cli/*.cpp)
add_executable(stochastic-windows-cli ${PROJECT_SOURCE_DIR}/src/stochastic-windows-cli/main-sw-dev.cpp ${STOCHASTIC_WINDOWS_CLI_SOURCES})
target_link_libraries(stochastic-windows-cli stochastic-windows storm storm-cli-utilities) # Adding headers for xcode
set_target_properties(stochastic-windows-cli PROPERTIES OUTPUT_NAME "storm-stochastic-windows")

add_dependencies(binaries stochastic-windows-cli)

# installation
install(TARGETS stochastic-windows-cli RUNTIME DESTINATION bin LIBRARY DESTINATION lib OPTIONAL)
