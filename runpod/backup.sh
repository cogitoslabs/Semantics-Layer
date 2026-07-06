#!/bin/bash
# 1. Capture the absolute windows path of the subfolder where this script lives
SUBFOLDER_PATH=$(pwd -W)

# 2. Move up to the parent directory to access the target project files
cd "$(dirname "$0")/.."

powershell -Command "
    \$subfolder = '${SUBFOLDER_PATH}'

    # --- 1. CLEANUP OLD BACKUPS ---
    if (Test-Path \"\$subfolder\project_backup.zip\") { Remove-Item \"\$subfolder\project_backup.zip\" -Force }
    if (Test-Path 'temp_archive_dir') { Remove-Item 'temp_archive_dir' -Recurse -Force }

    # --- 2. CREATE DESTINATION STRUCTURE ---
    New-Item -ItemType Directory -Path 'temp_archive_dir\data\raw' -Force | Out-Null

    # --- 3. COPY ITEMS ONE BY ONE ---
    # Copy the specific data subfolder
    if (Test-Path 'data\raw') { Copy-Item -Path 'data\raw\*' -Destination 'temp_archive_dir\data\raw' -Recurse -Force }

    # Copy the remaining project folders
    if (Test-Path 'evals')     { Copy-Item -Path 'evals'     -Destination 'temp_archive_dir' -Recurse -Force }
    if (Test-Path 'lib')       { Copy-Item -Path 'lib'       -Destination 'temp_archive_dir' -Recurse -Force }
    if (Test-Path 'notebooks') { Copy-Item -Path 'notebooks' -Destination 'temp_archive_dir' -Recurse -Force }
    if (Test-Path 'scripts')   { Copy-Item -Path 'scripts'   -Destination 'temp_archive_dir' -Recurse -Force }

    # Copy the individual configuration and code files
    if (Test-Path '.env')      { Copy-Item -Path '.env'      -Destination 'temp_archive_dir' -Force }
    if (Test-Path 'pipeline.py') { Copy-Item -Path 'pipeline.py' -Destination 'temp_archive_dir' -Force }

    # --- 4. COMPRESS & CLEANUP ---
    Compress-Archive -Path 'temp_archive_dir\*' -DestinationPath \"\$subfolder\project_backup.zip\" -Force
    Remove-Item 'temp_archive_dir' -Recurse -Force
"