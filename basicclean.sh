#!/bin/bash
#Script to automatically run basic cleaning algorithm, which involves CLFD and CoastGuard automatic cleaning
#as well as PAZI manual cleaning.
#Jacob Cardinal Tremblay

FILENAME=$1
NAME=${FILENAME%.*}
echo " "

#Run clfd on the file
echo "Step 1: CLFD"
singularity exec "/homes/jtremblay/fold-tools-2020-11-18-4cca94447feb.simg" clfd $FILENAME
echo " "

#Run CoastGuard on the file (python2.7). Save the output into a variable.
echo "Step 2: Coast_Guard"
cg_output="$(singularity exec "/homes/jtremblay/coastguard.sif" python /homes/jtremblay/clean.py -F rcvrstd -c 'resp=1290.05;6003.2,badfreqs=1450.0;1490.0' -F surgical -c 'cthresh=5,corder=1,cbp=None,cnp=4,sthresh=5,sorder=2;1,sbp=None,snp=2;4' ${FILENAME}.clfd)"

#Print coastguard output
echo $cg_output

#Save coastguard output file as a variable
cg_output_file=$(echo $cg_output | rev | cut -d " " -f1 | rev)
echo " "

#Manual Cleaning (press "p" after zapping to print to zapped channels to file, then press "s" to save)
#Press "q" to quit
echo "Step 3: Manual cleaning. Press "p" after zapping to print to zapped channels to file, then press "s" to save). Press "q" to quit."

touch ${NAME}_clean.txt

singularity exec "/homes/jtremblay/fold-tools-2020-11-18-4cca94447feb.simg" pazi $cg_output_file > ${NAME}_clean.txt
echo "Zapped channels are saved to ${NAME}_clean.txt."
echo "Cleaning Finished!"
echo " "

#Remove Files in between steps?
#rm ${FILENAME}.clfd
#rm ${FILENAME}_clfd_report.h5
#rm $cg_output_file
#rm ${NAME}_clean.txt

