@echo off
setlocal enabledelayedexpansion
if not exist build mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
copy /y Release\base58bruteforce.exe ..\base58bruteforce.exe
cd ..
