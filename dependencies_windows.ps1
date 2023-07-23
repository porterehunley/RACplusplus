$ErrorActionPreference = "Stop"

# Set CC and CXX environment variables to clang
$env:CC = (Get-Command gcc).Source
$env:CXX = (Get-Command g++).Source

# Download and unzip Eigen if it does not exist
if(-Not (Test-Path -Path "eigen-3.4.0")) {
    Invoke-WebRequest -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" -OutFile "eigen-3.4.0.zip"
    Expand-Archive -Path "eigen-3.4.0.zip" -DestinationPath "."
} else {
    Write-Output "Eigen already exists, skipping download."
}

$EIGEN_BUILD_DIR = "eigen-3.4.0\build"

if(Test-Path -Path $EIGEN_BUILD_DIR) {
    Write-Output "Directory $EIGEN_BUILD_DIR exists."
} else {
    # Create build directory
    New-Item -ItemType Directory -Force -Path $EIGEN_BUILD_DIR 
    Set-Location $EIGEN_BUILD_DIR 

    # Configure
    cmake -DCMAKE_C_COMPILER=$env:CC -DCMAKE_CXX_COMPILER=$env:CXX ..

    # Install
    make install
}

# Get the Python interpreter path
$PYTHON_PATH = (Get-Command python).Source

# Install pybind11 using the correct Python interpreter
& $PYTHON_PATH -m pip install pybind11

# Show pip install location for pybind
& $PYTHON_PATH -m pip show pybind11


# Print the environment variables
Write-Output "CC: $env:CC"
Write-Output "CXX: $env:CXX"
