#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Directory where Eigen should be installed
EIGEN_DIR=/usr/local/include/eigen3

# Check if Eigen is already installed
if [ ! -d "$EIGEN_DIR" ]; then
  # Download Eigen
  curl -OL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

  # Unzip Eigen
  unzip eigen-3.4.0.zip

  # Create build directory
  mkdir eigen-3.4.0/build
  cd eigen-3.4.0/build

  # Configure
  cmake ..

  # Install
  make install
else
  echo "Eigen is already installed at $EIGEN_DIR."
fi


# Get the Python interpreter path
PYTHON_PATH=$(which python)

# Install pybind11 using the correct Python interpreter
$PYTHON_PATH -m pip install pybind11
