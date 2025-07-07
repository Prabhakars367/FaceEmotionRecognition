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

# Python integration disabled - using C++ only implementation
set(PYTHON_INTEGRATION_ENABLED FALSE)

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

# Duplicate VideoEmotionAnalyzer.cpp has been permanently removed

# Python integration files have been permanently removed

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

# Also copy models to the output directory for the executable
add_custom_command(TARGET emotion_analyzer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    $<TARGET_FILE_DIR:emotion_analyzer>/models
    COMMENT "Copying models to executable directory"
)

# Copy required DLLs for Windows runtime
if(WIN32)
    # Copy OpenCV DLLs
    if(EXISTS "C:/opencv/build/x64/vc16/bin")
        add_custom_command(TARGET emotion_analyzer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/opencv/build/x64/vc16/bin/opencv_world4120.dll"
            "C:/opencv/build/x64/vc16/bin/opencv_world4120d.dll"
            "C:/opencv/build/x64/vc16/bin/opencv_videoio_ffmpeg4120_64.dll"
            $<TARGET_FILE_DIR:emotion_analyzer>
            COMMENT "Copying OpenCV DLLs to executable directory"
        )
    endif()
    
    # Copy PortAudio DLL if found
    if(PORTAUDIO_FOUND)
        get_filename_component(PORTAUDIO_DIR ${PORTAUDIO_LIBRARY} DIRECTORY)
        find_file(PORTAUDIO_DLL
            NAMES portaudio_x64.dll portaudio.dll
            PATHS
                "${PORTAUDIO_DIR}"
                "${PORTAUDIO_ROOT}/bin"
                "${PORTAUDIO_ROOT}/build/Release"
                "${PORTAUDIO_ROOT}/msvc/x64/Release"
            NO_DEFAULT_PATH
        )
        if(PORTAUDIO_DLL)
            add_custom_command(TARGET emotion_analyzer POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${PORTAUDIO_DLL}
                $<TARGET_FILE_DIR:emotion_analyzer>
                COMMENT "Copying PortAudio DLL to executable directory"
            )
        endif()
    endif()
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
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "============================")
message(STATUS "")
