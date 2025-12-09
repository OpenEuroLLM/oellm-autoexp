#!/bin/bash

##Clean up __pycache__ directories
##Fused kernels build directories and hipified cuda kernels
##Compiled data helpers file

find . -type d -name "__pycache__" -exec rm -rf {} +
