#!/bin/bash
#This is the main rfi-cleaning script. It will call the other scripts to do most of the cleaning.

total_files=$(find . -maxdepth 1 -name '*.ar' | wc -l)
processed_files=0

# Function to display a simple text-based progress bar
function display_progress() {
    local progress=$1
    local length=50  # Length of the progress bar

    # Calculate the number of characters to display based on the progress percentage
    local num_chars=$((progress * length / 100))
    
    # Print the progress bar
    printf "[%-*s] %d%%\r" "$length" "$(printf '%*s' "$num_chars" '=')" "$progress"

}

echo "Processing files:"

for FILENAME in *.ar; do
    NAME=${FILENAME}
    
    # Run the python script
    python rfi_statzap.py "$NAME"

    # Run the paz command to remove narrowband RFI found in pythong script
    paz -e .clean -k removed_freq_indexes.txt "$NAME"

    #Now we want to remove the old .ar extension to our NAME variable and replace it by the .clean extension
    CLEAN_NAME="${NAME%.*}.clean"    

    # Run the script which uses the paz command to remove wideband RFI found in pythong script
    ./bin_zapping.sh removed_bin_indexes.txt "$CLEAN_NAME"

    #Perform the lawn mowing cleaning from PSRCHIVE (algorithm that replaces spikey phase bins with the local median plus noise)
    #paz -e "${CLEAN_NAME}.L" -L $CLEAN_NAME
    paz -m -q -L $CLEAN_NAME
    
    # Clean up
    #rm removed_freq_indexes.txt
    #rm removed_bin_indexes.txt

    # Increment the processed files counter
    ((processed_files++))

    # Calculate the progress percentage
    progress=$((processed_files * 100 / total_files))

    # Display the progress bar
    display_progress $progress

    # echo "Number of cleaned files: "
    # ls -lR *.clean | wc -l
done

echo -e "\nProcessing complete."

#for FILENAME in *.ar; do
#	NAME=${FILENAME}
#	echo " "
#	
#	#We run the python script
#	python rfi_statzap.py $NAME
#	
#	paz -e .clean -k removed_indexes.txt $NAME
#	
#	rm removed_indexes.txt
#	
#	echo "Number of cleaned files: "
#	ls -lR *.clean | wc -l
#done
#echo "Cleaning Finished!"


