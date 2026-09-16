#!/bin/bash
# AWS OFI NCCL Installation Script
# This script installs AWS OFI NCCL with all required dependencies on Ubuntu systems
# Tested on Ubuntu 24.04 with CUDA 12.8

set -e  # Exit on error

echo "========================================="
echo "AWS OFI NCCL Installation Script"
echo "========================================="

# Step 1: Install libfabric development package (older version, will be upgraded)
echo "Step 1: Installing initial libfabric packages..."
sudo apt-get update
sudo apt-get install -y libfabric-dev

# Step 2: Install hwloc development package (required for aws-ofi-nccl)
echo "Step 2: Installing hwloc development package..."
sudo apt-get install -y libhwloc-dev

# Step 3: Download and install AWS EFA installer for newer libfabric (1.22.0+)
echo "Step 3: Downloading AWS EFA installer..."
cd /tmp
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz

echo "Installing AWS EFA with newer libfabric..."
cd aws-efa-installer
# Skip kernel module and limit configuration as they may not be needed on all instances
sudo ./efa_installer.sh -y --skip-kmod --skip-limit-conf

# Step 4: Clone aws-ofi-nccl repository
echo "Step 4: Cloning aws-ofi-nccl repository..."
cd /tmp
if [ -d "aws-ofi-nccl" ]; then
    echo "Removing existing aws-ofi-nccl directory..."
    rm -rf aws-ofi-nccl
fi
git clone https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl

# Step 5: Generate configure script
echo "Step 5: Generating configure script..."
autoreconf -ivf

# Step 6: Configure aws-ofi-nccl
echo "Step 6: Configuring aws-ofi-nccl build..."
# Detect CUDA installation path
CUDA_PATH=""
if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.8" ]; then
    CUDA_PATH="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12.7" ]; then
    CUDA_PATH="/usr/local/cuda-12.7"
else
    echo "Error: CUDA installation not found in standard locations"
    exit 1
fi

echo "Using CUDA installation at: $CUDA_PATH"

# Configure with AWS EFA paths
./configure \
    --with-libfabric=/opt/amazon/efa \
    --with-cuda=$CUDA_PATH \
    --with-mpi=/opt/amazon/openmpi \
    --enable-platform-aws

# Step 7: Build aws-ofi-nccl
echo "Step 7: Building aws-ofi-nccl..."
make -j$(nproc)

# Step 8: Install aws-ofi-nccl
echo "Step 8: Installing aws-ofi-nccl..."
sudo make install

# Step 9: Update library cache
echo "Step 9: Updating library cache..."
sudo ldconfig

# Step 10: Verify installation
echo "Step 10: Verifying installation..."
echo "Installed libraries:"
ls -la /usr/local/lib/libnccl* 2>/dev/null || echo "No NCCL libraries found in /usr/local/lib"

echo ""
echo "========================================="
echo "AWS OFI NCCL Installation Complete!"
echo "========================================="
echo ""
echo "The following libraries have been installed:"
echo "  - libnccl-net-ofi.so"
echo "  - libnccl-net.so"
echo "  - libnccl-ofi-tuner.so"
echo "  - libnccl-tuner-ofi.so"
echo ""
echo "Location: /usr/local/lib/"
echo ""
echo "To use AWS OFI NCCL with your applications, you may need to set:"
echo "  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
echo "  export NCCL_NET_PLUGIN=libnccl-net-ofi.so"
echo ""
echo "For EFA-enabled instances, also set:"
echo "  export FI_PROVIDER=efa"
echo "  export FI_EFA_USE_DEVICE_RDMA=1"
echo ""

# Cleanup
echo "Cleaning up temporary files..."
cd /
rm -rf /tmp/aws-efa-installer*
rm -rf /tmp/aws-ofi-nccl

echo "Installation script completed successfully!"