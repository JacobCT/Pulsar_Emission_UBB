#!/bin/bash

script_path="/srv/storage_11/galc/UBB/jtremblay/scripts/"

# Shuffle the list of filenames
shuffled_files=($(ls -1 *.ar | shuf))

# Set the maximum number of iterations
max_iterations=10

# Initialize counter
iteration_count=0

# Iterate over shuffled files
for FILENAME in "${shuffled_files[@]}"; do

	# Increment iteration count
    	((iteration_count++))

    	# Break out of the loop if maximum iterations reached
    	if [ "$iteration_count" -gt "$max_iterations" ]; then
        	break
    	fi

        NAME=${FILENAME}
        echo " "
	
	# Run the python script
    	python3 "$script_path"initial_statzap.py "$NAME"

	touch ${NAME}_clean.txt
	cat removed_freq_indexes.txt > "$NAME"_clean.txt

        echo "Zapped channels are saved to ${NAME}_clean.txt."
        find . -type f -empty -print -delete
        echo "Number of cleaned files: "
        ls -lR *_clean.txt | wc -l
done
echo "Initial cleaning done."
echo " "

