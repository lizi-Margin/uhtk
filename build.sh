#!/bin/zsh
rm -rf build
mkdir build && cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
&& make -j 12
cd ..

# cp ./build/c3utils/*.so ./c3utils
# cp ./build/c3utils/*.pyd ./c3utils
