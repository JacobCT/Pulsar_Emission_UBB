#!/bin/bash
#This script is to help in determining the frequency channels which will always be bad in an observation. It will automatically go through each file in a folder and allow you to clean the file manually. Once you reached the dsired number of files to clean, you can force stop the script and you will be left with .txt files that contain the list of all the bad frequencies in that observation.

for FILENAME in *.ar; do
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

