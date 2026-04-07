#!/bin/bash

# List of python scripts to execute
scripts=(
    "CatBoostBase.py"
    "XGBase.py"
    "TabPFNBase.py"
    "CatBoostRL.py"
    "TabPFNRL.py"
    "XGRL.py"
)

echo "Starting the Python execution pipeline..."

for script in "${scripts[@]}"; do
    if [[ -f "$script" ]]; then
        echo "--------------------------------------"
        echo "Running: $script"
        python "$script"
        
        # Check if the script exited with an error
        if [[ $? -ne 0 ]]; then
            echo "Error: $script failed to execute. Terminating pipeline."
            exit 1
        fi
    else
        echo "Warning: $script not found in the current directory. Skipping..."
    fi
done

echo "--------------------------------------"
echo "All scripts completed successfully!"