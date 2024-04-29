#!/bin/bash
#for FILENAME in *.ar; do

# Shuffle the list of filenames
shuffled_files=($(ls -1 *.ar | shuf))

# Iterate over shuffled files
for FILENAME in "${shuffled_files[@]}"; do
	NAME=${FILENAME}
	echo " "

	#Manual Cleaning (press "p" after zapping to print to zapped channels to file, then press "s" to save)
	#Press "q" to quit
	echo "Manual cleaning. Press "p" after zapping to print to zapped channels to file, then press "s" to save). Press "q" to quit."

	touch ${NAME}_clean.txt

	pazi $NAME > ${NAME}_clean.txt
	echo "Zapped channels are saved to ${NAME}_clean.txt."
	find . -type f -empty -print -delete
	echo "Number of cleaned files: "
	ls -lR *.txt | wc -l
done
echo "Cleaning Finished!"
echo " "

#Remove Files in between steps?
#rm ${FILENAME}.clfd
#rm ${FILENAME}_clfd_report.h5
#rm $cg_output_file
#rm ${NAME}_clean.txt

