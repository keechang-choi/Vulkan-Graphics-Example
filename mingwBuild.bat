if not exist build mkdir build
set arg1=%1
echo %arg1%
cd build
if "%arg1%" equ "Release" (
    set "arg1=-DCMAKE_BUILD_TYPE^=Release"
) else if "%arg1%" equ "Debug" (
    set "arg1=-DCMAKE_BUILD_TYPE^=Debug"
) else (
    set "arg1=-DCMAKE_BUILD_TYPE^=Debug"
)

REM cmake -S ../ -B . -G "MinGW Makefiles" %arg1%
REM mingw32-make.exe && mingw32-make.exe Shaders
cmake -S ../ -B . -G "Ninja" %arg1%
SETLOCAL
set NINJA_STATUS=[%%f/%%t(%%es)]: 
ninja.exe all && mingw32-make.exe Shaders
cd ..