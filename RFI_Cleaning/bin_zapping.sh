#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <bad_bin_list> <arfile>"
    exit 1
fi

input_file="$1"
data_file="$2"

#initialize array to store the -B terms
b_terms_array=()

# Read each line of the file
while IFS= read -r num; do
    if [ -z "$range_start" ]; then
        # If range_start is empty, set it to the current number
        range_start="$num"
        range_end="$num"
    elif [ "$((num - range_end))" -eq 1 ]; then
        # If the current number is consecutive to the previous one, update range_end
        range_end="$num"
    else
        # If not consecutive, add the -B term to the array
        b_terms_array+=("-B \"$range_start $range_end\"")

        # Reset variables for the next range
        range_start="$num"
        range_end="$num"
    fi
done < "$input_file"

# Add the last -B term to the array
b_terms_array+=("-B \"$range_start $range_end\"")

# Run the final command with all accumulated -B terms
final_command="paz -m -q ${b_terms_array[@]} $data_file"
#echo "Running command: $final_command"

# Uncomment the line below if you want to execute the final command
eval "$final_command"
