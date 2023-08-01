function(buildExample EXAMPLE_NAME)
    message(STATUS "Build Example: ${EXAMPLE_NAME}")
    set(EXAMPLE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME})
    set(MAIN_CPP ${EXAMPLE_FOLDER}/${EXAMPLE_NAME}.cpp)
    file(GLOB SOURCE
        *.cpp
        ${BASE_HEADERS}
        ${EXAMPLE_FOLDER}/*.cpp
        ${EXAMPLE_FOLDER}/*.hpp)
    add_executable(${EXAMPLE_NAME}
        ${MAIN_CPP}
        ${SOURCE})
    target_link_libraries(${EXAMPLE_NAME} base)

    set_property(TARGET ${EXAMPLE_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
endfunction(buildExample)

function(buildExamples)
    foreach(EXAMPLE ${EXAMPLES})
        buildExample(${EXAMPLE})
    endforeach(EXAMPLE)
endfunction(buildExamples)

# set(EXAMPLES
# triangle
# )
file(GLOB CHILDREN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*")

# list(FILTER EXAMPLES EXCLUDE REGEX ".*\\..*")
set(EXAMPLES "")

foreach(CHILD ${CHILDREN})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${CHILD})
        list(APPEND EXAMPLES ${CHILD})
    endif()
endforeach()

buildExamples()