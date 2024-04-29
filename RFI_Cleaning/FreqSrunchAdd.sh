#!/bin/bash

# Get the current directory where the script is located
directory="$(pwd)"

# Iterate over each file in the current directory with the ".clean" extension
for file in "$directory"/*.clean; do
    # Check if the file exists and is a regular file
    if [ -f "$file" ]; then
        # Execute your command on the file here
        #echo "Processing file: $file"
        # Example command: replace 'your_command' with your actual command
        pam -F -e Fscrunch $file
    fi
done

psradd -o added.FF *.Fscrunch
rm *.Fscrunch
