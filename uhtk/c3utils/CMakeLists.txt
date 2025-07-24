message("Enter subdirectory c3utils")

# include_directories(external/Eigen)
# add_subdirectory(external/pybind11)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

add_executable(test_c3utils "test_c3utils.cpp" "vector.h" "funcs.h" "c3utils.h" "def.h" "io.h")
# set_target_properties(test_c3utils PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )

set(Python3_EXECUTABLE "/usr/bin/python3.13")
set(Python3_INCLUDE_DIR "/usr/include/python3.13")
set(Python3_LIBRARY "/usr/lib/libpython3.13.so.1.0")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
pybind11_add_module(py_c3utils_native PyInterface.cpp)
target_link_libraries(py_c3utils_native PRIVATE pybind11::embed)

function(setup_pybind_proj_win pybind_proj_name conda_env_path conda_env_python_version)
    if(NOT EXISTS "${conda_env_path}/python.exe")
        message(FATAL_ERROR "Python executable not found at ${conda_env_path}/python.exe")
    endif()
    set(Python3_EXECUTABLE "${conda_env_path}/python.exe")
	set(Python3_INCLUDE_DIR "${conda_env_path}/include")
	set(Python3_LIBRARY "${conda_env_path}/libs/python${conda_env_python_version}.lib")
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Python3 development files not found")
    endif()

	set(pybind_proj_name_ "${pybind_proj_name}.cp${conda_env_python_version}-win_amd64")
	pybind11_add_module(${pybind_proj_name_} PyInterface.cpp)
	target_include_directories(${pybind_proj_name_} PRIVATE ${Python3_INCLUDE_DIRS})
	target_link_libraries(${pybind_proj_name_} PRIVATE ${Python3_LIBRARIES})
	
	set_target_properties(${pybind_proj_name_} PROPERTIES
		PREFIX ""  
		SUFFIX ".pyd"      
		LINKER_LANGUAGE CXX  
    	# RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
	)
endfunction()


function(setup_pybind_proj_linux pybind_proj_name conda_env_path conda_env_python_version conda_env_python_version_dot)
    if(NOT EXISTS "${conda_env_path}/bin/python${conda_env_python_version_dot}")
        message(FATAL_ERROR "Python executable not found at ${conda_env_path}/bin/python${conda_env_python_version_dot}")
    endif()
    set(Python3_EXECUTABLE "${conda_env_path}/bin/python${conda_env_python_version_dot}")
	set(Python3_INCLUDE_DIR "${conda_env_path}/include/python${conda_env_python_version_dot}/")
	set(Python3_LIBRARY "${conda_env_path}/lib/libpython${conda_env_python_version_dot}.so.1.0")
	find_package(Python3 REQUIRED   COMPONENTS Interpreter Development)
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Python3 development files not found")
    endif()

	set(pybind_proj_name_ "${pybind_proj_name}.cpython-${conda_env_python_version}-x86_64-linux-gnu")
	pybind11_add_module(${pybind_proj_name_} PyInterface.cpp)
	target_include_directories(${pybind_proj_name_} PRIVATE ${Python3_INCLUDE_DIRS})
	target_link_libraries(${pybind_proj_name_} PRIVATE ${Python3_LIBRARIES})
	
	set_target_properties(${pybind_proj_name_} PROPERTIES
		PREFIX ""  
		SUFFIX ".so"      
		LINKER_LANGUAGE CXX  
    	# RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
	)
endfunction()

setup_pybind_proj_linux("py_c3utils" "/home/hulc/.conda/envs/hmp" "312" "3.12")
setup_pybind_proj_linux("py_c3utils" "/home/hulc/.conda/envs/label" "311" "3.11")
setup_pybind_proj_linux("py_c3utils" "/home/hulc/.conda/envs/RVC" "39" "3.9")
# setup_pybind_proj_linux("py_c3utils" "/home/hulc/.conda/envs/old" "37" "3.7m")
