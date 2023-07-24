if not exist build mkdir build
set arg1=%1
cd build
cmake -S ../ -B . -G "MinGW Makefiles" %arg1%
mingw32-make.exe && mingw32-make.exe Shaders
cd ..