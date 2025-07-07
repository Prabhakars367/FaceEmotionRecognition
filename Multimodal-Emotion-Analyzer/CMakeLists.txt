cmake_minimum_required(VERSION 3.20)
project(MultimodalEmotionAnalyzer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build test programs" OFF)
option(USE_SYSTEM_OPENCV "Use system OpenCV" ON)
option(NO_AUDIO "Disable audio processing" OFF)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect imgcodecs videoio)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and set OpenCV_DIR correctly.")
endif()

# Python and pybind11 integration (optional)
option(ENABLE_PYTHON_INTEGRATION "Enable Python-based deep learning inference" ON)

if(ENABLE_PYTHON_INTEGRATION)
    # Find Python
    find_package(Python COMPONENTS Interpreter Development QUIET)
    if(Python_FOUND)
        message(STATUS "Found Python: ${Python_EXECUTABLE} (version ${Python_VERSION})")
        
        # Try to find pybind11
        find_package(pybind11 QUIET)
        if(pybind11_FOUND)
            message(STATUS "Found pybind11: ${pybind11_VERSION}")
            set(PYTHON_INTEGRATION_ENABLED TRUE)
        else()
            # Try to find pybind11 via pip install
            execute_process(
                COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE pybind11_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(pybind11_DIR)
                find_package(pybind11 PATHS ${pybind11_DIR} QUIET)
                if(pybind11_FOUND)
                    message(STATUS "Found pybind11 via Python: ${pybind11_VERSION}")
                    set(PYTHON_INTEGRATION_ENABLED TRUE)
                endif()
            endif()
        endif()
    endif()
    
    if(NOT PYTHON_INTEGRATION_ENABLED)
        message(WARNING "Python or pybind11 not found. Python integration will be disabled.")
        message(STATUS "To enable Python integration, install: pip install pybind11")
        set(PYTHON_INTEGRATION_ENABLED FALSE)
    endif()
else()
    set(PYTHON_INTEGRATION_ENABLED FALSE)
    message(STATUS "Python integration disabled by user option")
endif()

# Set OpenCV path (Windows specific - adjust as needed)
if(WIN32)
    set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
endif()

# For Windows/MSVC, find PortAudio using your specific installation
if(WIN32 AND NOT NO_AUDIO)
    set(PORTAUDIO_ROOT "C:/Users/KIIT/Downloads/pa_stable_v190700_20210406 (1)/portaudio")
    
    find_path(PORTAUDIO_INCLUDE_DIR portaudio.h
        PATHS
            "${PORTAUDIO_ROOT}/include"
        NO_DEFAULT_PATH
    )
    
    find_library(PORTAUDIO_LIBRARY
        NAMES portaudio_x64 portaudio_static_x64 portaudio
        PATHS
            "${PORTAUDIO_ROOT}/lib/x64/Release"
            "${PORTAUDIO_ROOT}/lib/Release"
            "${PORTAUDIO_ROOT}/build/Release"
            "${PORTAUDIO_ROOT}/msvc/x64/Release"
        NO_DEFAULT_PATH
    )
    
    if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARY)
        set(PORTAUDIO_FOUND TRUE)
        set(PORTAUDIO_INCLUDE_DIRS ${PORTAUDIO_INCLUDE_DIR})
        set(PORTAUDIO_LIBRARIES ${PORTAUDIO_LIBRARY})
        message(STATUS "Found PortAudio: ${PORTAUDIO_LIBRARY}")
    else()
        message(WARNING "PortAudio not found in ${PORTAUDIO_ROOT}")
        set(PORTAUDIO_FOUND FALSE)
    endif()
else()
    # For Unix systems, use pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PORTAUDIO portaudio-2.0)
        if(PORTAUDIO_FOUND)
            message(STATUS "Found PortAudio via pkg-config")
        else()
            message(WARNING "PortAudio not found via pkg-config")
            set(PORTAUDIO_FOUND FALSE)
        endif()
    else()
        message(WARNING "pkg-config not found, PortAudio detection skipped")
        set(PORTAUDIO_FOUND FALSE)
    endif()
endif()

# Check if PortAudio was found or if NO_AUDIO is set
if(NOT PORTAUDIO_FOUND OR NO_AUDIO)
    message(WARNING "PortAudio not found or disabled. Audio processing will be disabled.")
    add_definitions(-DNO_AUDIO)
    set(PORTAUDIO_LIBRARIES "")
    set(PORTAUDIO_INCLUDE_DIRS "")
endif()

# Source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.hpp"
)

# Remove duplicate VideoEmotionAnalyzer.cpp to avoid linking conflicts
set(DUPLICATE_FILE "${CMAKE_SOURCE_DIR}/src/VideoEmotionAnalyzer.cpp")
if(EXISTS ${DUPLICATE_FILE})
    list(REMOVE_ITEM SOURCES ${DUPLICATE_FILE})
    message(STATUS "Excluded duplicate file: ${DUPLICATE_FILE}")
endif()

# Create executable
if(NOT SOURCES)
    message(FATAL_ERROR "No source files found!")
endif()

add_executable(emotion_analyzer ${SOURCES})

# Set target properties
set_target_properties(emotion_analyzer PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(emotion_analyzer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Include PortAudio directories only if found
if(PORTAUDIO_FOUND)
    target_include_directories(emotion_analyzer PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
endif()

# Include Python directories if integration is enabled
if(PYTHON_INTEGRATION_ENABLED)
    target_include_directories(emotion_analyzer PRIVATE ${Python_INCLUDE_DIRS})
    target_compile_definitions(emotion_analyzer PRIVATE PYTHON_INTEGRATION_ENABLED)
endif()

# Link libraries
target_link_libraries(emotion_analyzer 
    PRIVATE ${OpenCV_LIBS}
)

# Link PortAudio only if found
if(PORTAUDIO_FOUND)
    target_link_libraries(emotion_analyzer PRIVATE ${PORTAUDIO_LIBRARIES})
    message(STATUS "Linking PortAudio: ${PORTAUDIO_LIBRARIES}")
else()
    message(STATUS "Building without PortAudio support")
endif()

# Link Python libraries if integration is enabled
if(PYTHON_INTEGRATION_ENABLED)
    target_link_libraries(emotion_analyzer PRIVATE pybind11::embed)
    message(STATUS "Linking Python: ${Python_LIBRARIES}")
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(emotion_analyzer PRIVATE 
        /W4                    # High warning level
        /permissive-           # Disable non-conforming code
        /Zc:__cplusplus        # Enable correct __cplusplus macro
    )
    # Disable specific warnings that are too noisy
    target_compile_options(emotion_analyzer PRIVATE 
        /wd4244                # Disable conversion warnings (double to float)
        /wd4100                # Disable unreferenced parameter warnings
    )
else()
    target_compile_options(emotion_analyzer PRIVATE 
        -Wall -Wextra -pedantic
        -Wno-unused-parameter  # Disable unused parameter warnings
    )
endif()

# Copy models directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})

# Copy Python scripts to build directory if Python integration is enabled
if(PYTHON_INTEGRATION_ENABLED)
    file(COPY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_BINARY_DIR})
endif()

# Also copy models to the output directory for the executable
add_custom_command(TARGET emotion_analyzer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    $<TARGET_FILE_DIR:emotion_analyzer>/models
    COMMENT "Copying models to executable directory"
)

# Copy Python scripts to output directory if Python integration is enabled
if(PYTHON_INTEGRATION_ENABLED)
    add_custom_command(TARGET emotion_analyzer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python
        $<TARGET_FILE_DIR:emotion_analyzer>/python
        COMMENT "Copying Python scripts to executable directory"
    )
endif()

# Install targets
install(TARGETS emotion_analyzer 
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)
install(DIRECTORY models/ 
    DESTINATION share/emotion_analyzer/models
    COMPONENT Runtime
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "PortAudio found: ${PORTAUDIO_FOUND}")
if(PORTAUDIO_FOUND)
    message(STATUS "PortAudio libraries: ${PORTAUDIO_LIBRARIES}")
endif()
message(STATUS "Audio support: ${PORTAUDIO_FOUND}")
message(STATUS "Python integration: ${PYTHON_INTEGRATION_ENABLED}")
if(PYTHON_INTEGRATION_ENABLED)
    message(STATUS "Python version: ${Python_VERSION}")
    message(STATUS "Python executable: ${Python_EXECUTABLE}")
    message(STATUS "pybind11 version: ${pybind11_VERSION}")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "============================")
message(STATUS "")
