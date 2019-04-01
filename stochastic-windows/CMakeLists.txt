file(GLOB_RECURSE ALL_FILES ${PROJECT_SOURCE_DIR}/src/stochastic-windows/*.h ${PROJECT_SOURCE_DIR}/src/stochastic-windows/*.cpp)

register_source_groups_from_filestructure("${ALL_FILES}" stochastic-windows)



file(GLOB_RECURSE STOCHASTIC_WINDOWS_SOURCES ${PROJECT_SOURCE_DIR}/src/stochastic-windows/*.cpp)
file(GLOB_RECURSE STOCHASTIC_WINDOWS_HEADERS ${PROJECT_SOURCE_DIR}/src/stochastic-windows/*.h)


# Create stochastic-windows.
add_library(stochastic-windows SHARED ${STOCHASTIC_WINDOWS_SOURCES} ${STOCHASTIC_WINDOWS_HEADERS})

# Remove define symbol for shared libstorm.
set_target_properties(stochastic-windows PROPERTIES DEFINE_SYMBOL "")
#add_dependencies(storm resources)
list(APPEND STORM_TARGETS stochastic-windows)
set(STORM_TARGETS ${STORM_TARGETS} PARENT_SCOPE)

target_link_libraries(stochastic-windows PUBLIC storm storm-parsers storm-conv ${STORM_STOCHASTIS_WINDOWS_LINK_LIBRARIES})

# Install storm headers to include directory.
foreach(HEADER ${STOCHASTIC_WINDOWS_HEADERS})
    string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/src/?" "" RELATIVE_HEADER_PATH ${HEADER})
    string(REGEX MATCH "(.*)[/\\]" RELATIVE_DIRECTORY ${RELATIVE_HEADER_PATH})
    string(REGEX REPLACE "${RELATIVE_DIRECTORY}/?" "" HEADER_FILENAME ${RELATIVE_HEADER_PATH})
    add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/include/${RELATIVE_DIRECTORY}${HEADER_FILENAME}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include/${RELATIVE_DIRECTORY}
            COMMAND ${CMAKE_COMMAND} -E copy ${HEADER} ${CMAKE_BINARY_DIR}/include/${RELATIVE_DIRECTORY}${HEADER_FILENAME}
            DEPENDS ${HEADER}
    )
    list(APPEND STOCHASTIC_WINDOWS_OUTPUT_HEADERS "${CMAKE_BINARY_DIR}/include/${RELATIVE_DIRECTORY}${HEADER_FILENAME}")
endforeach()
add_custom_target(copy_stochastic_windows_headers DEPENDS ${STOCHASTIC_WINDOWS_OUTPUT_HEADERS} ${STOCHASTIC_WINDOWS_HEADERS})
add_dependencies(stochastic-windows copy_stochastic_windows_headers)

# installation
install(TARGETS stochastic-windows RUNTIME DESTINATION bin LIBRARY DESTINATION lib OPTIONAL)

