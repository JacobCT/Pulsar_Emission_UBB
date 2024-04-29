#!/bin/bash

# Directory containing the files
directory="."

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Directory does not exist."
    exit 1
fi

# Pattern to match files ending with _clean.txt
pattern="*_clean.txt"

# Use awk to count occurrences of each number in all files and filter numbers that occur in every file
awk '
    FNR == 1 { ++fileCount }
    /^[0-9]+$/ { counts[$0]++ }
    END {
        for (num in counts) {
            if (counts[num] == fileCount) {
                print num
            }
        }
    }
' $pattern | sort -n > zapped_freqs.txt

echo "Result written to zapped_freqs.txt"
