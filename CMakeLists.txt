cmake_minimum_required(VERSION 3.18)
project(base58bruteforce VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 17)

# ---------------- sources ----------------
set(DEC_SRC src/main.cu)
list(REMOVE_DUPLICATES DEC_SRC)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${DEC_SRC})

# -------------- target -------------------
add_executable(base58bruteforce ${DEC_SRC})

target_include_directories(base58bruteforce PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_compile_features(base58bruteforce PRIVATE cxx_std_17)

target_compile_options(base58bruteforce PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo -Xptxas -O3>
    $<$<AND:$<NOT:$<COMPILE_LANGUAGE:CUDA>>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native>
)

if(WIN32)
    target_link_libraries(base58bruteforce PRIVATE advapi32)
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(base58bruteforce PRIVATE Threads::Threads)
endif()

install(TARGETS base58bruteforce DESTINATION bin)
install(EXPORT base58bruteforceTargets
    FILE base58bruteforceTargets.cmake
    NAMESPACE base58bruteforce::
    DESTINATION lib/cmake/base58bruteforce
)
