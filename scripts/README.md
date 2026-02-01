# Scripts Directory

This directory contains all build, setup, utility, and CI scripts organized by type and platform.

## Directory Structure

```
scripts/
├── build/           # Build scripts
│   ├── sh/         # Shell build scripts (macOS/Linux)
│   ├── bat/        # Batch build scripts (Windows)
│   └── py/         # Python build scripts
├── setup/          # Environment setup scripts
├── utils/          # Utility and check scripts
├── ci/             # CI and test scripts
└── README.md       # This file
```

## Build Scripts

### Shell Scripts (`build/sh/`)

- **`build.sh`** - Main build script (platform-agnostic)
- **`build-intel.sh`** - Intel Mac specific build
- **`build_manual_mac.sh`** - Manual Mac build script
- **`build_runtime_mac_intel.sh`** - Build runtime for Intel Mac
- **`build_runtime_mac_mps.sh`** - Build runtime for Mac MPS (ARM)

### Batch Scripts (`build/bat/`)

- **`build_manual.bat`** - Manual Windows build script
- **`build_runtime.bat`** - Build runtime for Windows

### Python Scripts (`build/py/`)

- **`build_dsu.py`** - Main DSU build script (cx_Freeze)
- **`build_dsu_launchers.py`** - Build launcher executables

## Setup Scripts (`setup/`)

- **`setup_build_env.bat`** - Setup build environment on Windows
- **`setup_local_mac.sh`** - Setup local environment on Mac

## Utility Scripts (`utils/`)

- **`check_demucs_save.py`** - Check Demucs save functionality
- **`check_wav.py`** - Check WAV file utilities
- **`profile_worker.py`** - Profile worker performance

## CI Scripts (`ci/`)

- **`test_build_env.bat`** - Test build environment on Windows
- **`simulate_ci_mac_arm.sh`** - Simulate CI on Mac ARM
- **`simulate_ci_real_audio.py`** - Simulate CI with real audio

## Other Scripts

- **`run_local.sh`** - Run local development environment

## Usage

### Building

**Mac/Linux:**
```bash
cd scripts/build/sh
./build.sh
./build-intel.sh
./build_runtime_mac_intel.sh
```

**Windows:**
```cmd
cd scripts\build\bat
build_manual.bat
build_runtime.bat
```

**Python (all platforms):**
```bash
cd scripts/build/py
python build_dsu.py
python build_dsu_launchers.py
```

### Setup

**Mac:**
```bash
cd scripts/setup
./setup_local_mac.sh
```

**Windows:**
```cmd
cd scripts\setup
setup_build_env.bat
```

### Utilities

```bash
cd scripts/utils
python check_wav.py
python profile_worker.py
```

## Notes

- All scripts should be run from the project root directory
- Build scripts may require the runtime environment to be set up first
- Some scripts may have platform-specific requirements
