cmake_minimum_required(VERSION 3.24)

project(lib_src VERSION 0.0.1 DESCRIPTION "The core library of the MachineLearning library.")
set(LIBRARY_TARGET_NAME MLCore)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_compile_definitions(MACHINE_LEARNING_EXPORTS)


option(ML_SHARED_BUILD "Use the MachineLearning library in shared mode." ON)

set(ML_SRC_FILES
        include/ext.h
        src/ext.cpp
        src/ranges.cpp
        src/ranges.hpp
        # TODO: Remove main.cpp
        src/main.cpp
        src/Models/Linear.cpp
        src/Models/Linear.hpp
        src/Models/MLP.cpp
        src/Models/MLP.hpp
)

if(ML_SHARED_BUILD)
    add_library(${LIBRARY_TARGET_NAME} SHARED
            ${ML_SRC_FILES}
    )
else()
    add_library(${LIBRARY_TARGET_NAME}
            ${ML_SRC_FILES}
    )
endif ()

if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
    add_compile_definitions(ML_DEBUG)
endif()

set_property(TARGET ${LIBRARY_TARGET_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

include(GenerateExportHeader)
generate_export_header(${LIBRARY_TARGET_NAME}
        BASE_NAME "MachineLearning"
        EXPORT_MACRO_NAME "ML_API"
        EXPORT_FILE_NAME "export/ml_export.hpp"
)
target_include_directories(${LIBRARY_TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/export)

add_library(${PROJECT_NAME}::${LIBRARY_TARGET_NAME} ALIAS ${LIBRARY_TARGET_NAME})

set_target_properties(${LIBRARY_TARGET_NAME} PROPERTIES
        VERSION       ${${PROJECT_NAME}_VERSION}
        PUBLIC_HEADER "include/MachineLearning/MachineLearning.hpp"
        PUBLIC cxx_std_20
)

target_include_directories(${LIBRARY_TARGET_NAME} PUBLIC include)
target_include_directories(${LIBRARY_TARGET_NAME} PRIVATE src)

#target_link_libraries(${LIBRARY_TARGET_NAME} PUBLIC glm)

# Specify installation targets, typology and destination folders.
install(TARGETS ${LIBRARY_TARGET_NAME}
        EXPORT  ${PROJECT_NAME}
        LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                            COMPONENT shlib
        ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                            COMPONENT lib
        RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}"                            COMPONENT bin
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_TARGET_NAME}" COMPONENT dev)


message(STATUS "Created target ${LIBRARY_TARGET_NAME} for export ${PROJECT_NAME}.")

# Also make an executable for testing purposes (memleaks)
# TODO: Remove
add_executable(${PROJECT_NAME}_test
        src/main.cpp
        src/Models/Linear.cpp
        src/Models/Linear.hpp
        src/Models/MLP.cpp
        src/Models/MLP.hpp
        src/ranges.cpp
        src/ranges.hpp
        #src/ext.cpp
        #src/ext.hpp
        )
