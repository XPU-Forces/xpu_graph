cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(super_kernel_launcher LANGUAGES CXX)
# WARNING(liuyuan):
# DEFINED $ENV{ASCEND_HOME_PATH} DOES NOT work under PIP ioslation building.
set(ASCEND_SDK_PATH /usr/local/Ascend)
if (DEFINED $ENV{ASCEND_HOME_PATH})
  set(ASCEND_SDK_PATH $ENV(ASCEND_HOME_PATH))
endif()

if (NOT EXISTS ${ASCEND_SDK_PATH})
  message(WARNING "This is not an environment with Ascend SDK.")
  return()
endif()

execute_process(COMMAND bash -c "which c++" OUTPUT_VARIABLE CXX_PATH RESULT_VARIABLE _RET_CODE)
if (NOT _RET_CODE MATCHES 0)
  message(FATAL_ERROR "NO c++ found.")
endif()
string(STRIP ${CXX_PATH} CXX_PATH)
set(CMAKE_CXX_COMPILER ${CXX_PATH})

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/runtime)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/runtime)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/runtime)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)

find_program(GOLD_FOUND gold)
if(LLD_FOUND)
  add_link_options(-fuse-ld=lld -L/usr/local/lib)
elseif(GOLD_FOUND)
  add_link_options(-fuse-ld=gold -L/usr/local/lib)
endif()



if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_INSTALL_PATH "/usr/local/Ascend/latest")
else()
  set(ASCEND_INSTALL_PATH "$ENV{ASCEND_HOME_PATH}")
endif()

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# WARNING(liuyuan): If you're using uv and try to [uv pip install -e .], don't forget the [--no-build-isolation] for the corrent virtual environment.
#                   uv pip install -e . --no-build-isolation
execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import os; import torch; \
  print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _RET_CODE
  OUTPUT_VARIABLE TORCH_DIR)
if (NOT _RET_CODE MATCHES 0)
  message(AUTHOR_WARNING "If you're using [pip install -e .] under virtualenv and uv, don't forget the [--no-build-isolation] for the corrent virtual environment.")
  message(FATAL_ERROR "NO Torch found.")
endif()
set(Torch_DIR ${TORCH_DIR}/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import os; import torch_npu; \
  print(os.path.dirname(torch_npu.__file__),end='');"
  RESULT_VARIABLE _RET_CODE
  OUTPUT_VARIABLE TORCH_NPU_DIR)
if (NOT _RET_CODE MATCHES 0)
  message(FATAL_ERROR "NO Torch_NPU found.")
endif()


configure_file(super_kernel_launcher.cpp.in ${CMAKE_BINARY_DIR}/src/super_kernel_launcher.cpp @ONLY)
pybind11_add_module(${CMAKE_PROJECT_NAME} SHARED ${CMAKE_BINARY_DIR}/src/super_kernel_launcher.cpp)
target_link_options(${CMAKE_PROJECT_NAME} INTERFACE "-Wl,--no-undefined")
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${ASCEND_INSTALL_PATH}/runtime/include/ ${TORCH_NPU_DIR}/include)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${TORCH_NPU_LIBRARY} ${ASCEND_INSTALL_PATH}/runtime/lib64/libascendcl.so pybind11::headers)
