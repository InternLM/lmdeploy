# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/thewh1teagle/vibe/blob/5d7b75568ca65ab635bdf0ce912bbc975a043066/scripts/setup_cuda.ps1

$CUDA_VERSION_FULL = $env:INPUT_CUDA_VERSION # v12.1.0 or v11.8.0

# Make sure CUDA_VERSION_FULL is set and valid, otherwise error.
# Validate CUDA version, extracting components via regex
$cuda_ver_matched = $CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)\.(?<patch>[0-9]+)$"
if(-not $cuda_ver_matched){
    Write-Output "Invalid CUDA version specified, <major>.<minor>.<patch> required. '$CUDA_VERSION_FULL'."
    exit 1
}
$CUDA_MAJOR=$Matches.major
$CUDA_MINOR=$Matches.minor
$CUDA_PATCH=$Matches.patch

Write-Output "Selected CUDA version: $CUDA_VERSION_FULL"

$src = "cuda"
$dst = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($CUDA_MAJOR).$($CUDA_MINOR)"
$installer = "cuda.exe"

if ($CUDA_VERSION_FULL -eq "12.1.0") {
    $downloadUrl = "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe"
} elseif ($CUDA_VERSION_FULL -eq "11.8.0") {
    $downloadUrl = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe"
} elseif ($CUDA_VERSION_FULL -eq "12.5.0") {
    $downloadUrl = "https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.85_windows.exe"
} elseif ($CUDA_VERSION_FULL -eq "12.6.2") {
    $downloadUrl = "https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.94_windows.exe"
} elseif ($CUDA_VERSION_FULL -eq "12.8.1") {
    $downloadUrl = "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe"
} else {
    Write-Output "Unsupported CUDA version specified"
    exit 1
}

# Download cuda
Write-Output "Downloading CUDA from: $downloadUrl"
if (-not (Test-Path -Path $installer)) {
    Write-Output "Downloading CUDA installer..."
    # If the file does not exist, download it
    & "C:\msys64\usr\bin\wget" $downloadUrl -O $installer -q
}

# Extract cuda
if (-not (Test-Path -Path $src -Type Container)) {
    # Extract CUDA using 7-Zip
    Write-Output "Extracting CUDA using 7-Zip..."
    mkdir "$src"
    & 'C:\Program Files\7-Zip\7z' x $installer -o"$src"
}

# Create destination directory if it doesn't exist
if (-Not (Test-Path -Path $dst)) {
    Write-Output "Creating destination directory: $dst"
    New-Item -Path $dst -ItemType Directory
}

# Get directories to process from the source path
$directories = Get-ChildItem -Directory -Path $src
$whitelist = @("CUDA_Toolkit_Release_Notes.txt", "DOCS", "EULA.txt", "LICENSE", "README", "version.json")

foreach ($dir in $directories) {
    # Get all subdirectories and files in the current directory
    $items = Get-ChildItem -Path (Join-Path $src $dir.Name)

    foreach ($item in $items) {
        if ($item.PSIsContainer) {
            # If the item is a directory, copy its contents
            Write-Output "Copying contents of directory $($item.FullName) to $dst"
            Copy-Item -Path "$($item.FullName)\*" -Destination $dst -Recurse -Force
        } else {
            if ($whitelist -contains $item.Name) {
                Write-Output "Copying file $($item.FullName) to $dst"
                Copy-Item -Path $item.FullName -Destination $dst -Force
            }
        }
    }
}

# Add msbuild cuda extensions
$msBuildExtensions = (Get-ChildItem  "$src\visual_studio_integration\CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions").fullname
(Get-ChildItem 'C:\Program Files\Microsoft Visual Studio\2022\*\MSBuild\Microsoft\VC\*\BuildCustomizations').FullName | ForEach-Object {
    $destination = $_
    $msBuildExtensions | ForEach-Object {
        $extension = $_
        Copy-Item $extension -Destination $destination -Force
        Write-Output "Copied $extension to $destination"
    }
}

$CUDA_FLAGS="-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH=1"

# Add to Github env
Write-Output "Setting environment variables for GitHub Actions..."

Write-Output "CUDA_PATH=$dst"
Write-Output "CUDA_PATH_V$($CUDA_MAJOR)_$($CUDA_MINOR)=$dst"
Write-Output "CUDA_PATH_VX_Y=CUDA_PATH_V$($CUDA_MAJOR)_$($CUDA_MINOR)"
Write-Output "CUDA_VERSION=$CUDA_VERSION_FULL"

Write-Output "CUDA_PATH=$dst" >> $env:GITHUB_ENV
Write-Output "CUDA_PATH_V$($CUDA_MAJOR)_$($CUDA_MINOR)=$dst" >> $env:GITHUB_ENV
Write-Output "CUDA_PATH_VX_Y=CUDA_PATH_V$($CUDA_MAJOR)_$($CUDA_MINOR)" >> $env:GITHUB_ENV
Write-Output "CudaToolkitDir=$dst" >> $env:GITHUB_ENV
Write-Output "CMAKE_CUDA_COMPILER=$dst\bin\nvcc.exe" >> $env:GITHUB_ENV
Write-Output "NVCC_APPEND_FLAGS=$CUDA_FLAGS" >> $env:GITHUB_ENV

Write-Output "CUDA_VERSION=$CUDA_VERSION_FULL" >> $env:GITHUB_ENV
Write-Output "Setup completed."
