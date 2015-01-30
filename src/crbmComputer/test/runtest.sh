#!/bin/bash

#export LD_LIBRARY_PATH=/home/orionx/crbm/build/src/crbmComputer
export LD_LIBRARY_PATH=@CMAKE_CURRENT_BINARY_DIR@
export PYTHONPATH=@CMAKE_CURRENT_SOURCE_DIR@:${PYTHONPATH}


python @CMAKE_CURRENT_SOURCE_DIR@/test/crbmComputerTest.py @CMAKE_CURRENT_SOURCE_DIR@/test/mycka.jpg
