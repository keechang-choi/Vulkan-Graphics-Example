if not exist build mkdir build
set arg1=%1
echo %arg1%
cd build
if "%arg1%" equ "" (
    set "arg1=-DCMAKE_BUILD_TYPE^=Release"
) else if "%arg1%" equ "Debug" (
    set "arg1=-DCMAKE_BUILD_TYPE^=Debug"
)
cmake -S ../ -B . -G "MinGW Makefiles" %arg1%
mingw32-make.exe && mingw32-make.exe Shaders
cd ..