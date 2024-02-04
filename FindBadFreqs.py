#Given a folder full of RFI-zapped PSRCHIVE archive files, will return a list of frequencies zaps in all observations.
#Jacob Cardinal Tremblay

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from functools import reduce
import os, glob
import psrchive

def convert_file_to_int(file):
    templist = []
    #Read in the numbers between the quotes
    quotes = re.findall(r'"[^"]*"', file)
    #Make a list of what is in the quotes
    for quote in quotes:
        templist.append(quote)
    
    #Split each number into it's own value of the array
    #splitarr = quotes.split()
    #Remove the quotations
    #splitarr = [s.replace('\"', '') for s in splitarr]
    #Convert the strings to ints
    #intarr = [eval(i) for i in splitarr]
    
    intarr = []
    for quote in templist:
        # Remove the quotations
        clean_quote = quote.replace('"', '')
        # Split each number into its own value of the array
        splitarr = clean_quote.split()
        # Convert the strings to ints
        intarr.extend([eval(i) for i in splitarr])
    
    
    return(intarr)

def chn_to_freq(path, file):
    #Load in the raw data and transform it
    full_path = path + file
    File = psrchive.Archive_load(full_path)
    File.tscrunch()
    File.dedisperse()
    File.pscrunch()
    File.remove_baseline()
    freq_mhz_table = File.get_frequency_table()
    freq_mhz = freq_mhz_table[0]
    return(freq_mhz)

def export_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

#Create the arrays we will use
file_list = []
freq_list = []
#Define the path of the file
#We can either set the path:
#path = '/srv/storage_11/galc/UBB/jtremblay/20231227/singlepulse/search2/'
#Or we can use the current directory:
current_directory = os.getcwd()
path = str(current_directory) + "/"
#print("File path:", path)

#Acess the directory
files = os.listdir(path)

#Look into the directory and find a file ending in pazi, which will then give the frequencies
for file in files:
    if file.endswith('.pazi'):
        freqs = chn_to_freq(path, file)
        freq_list.append(freqs)
        break


#Find the manually cleaned files and get a list of the bad indexes
for filename in glob.glob(os.path.join(path, '*_clean.txt')):
    f = open(filename, 'r').read()
    file  = convert_file_to_int(f)
    file_list.append(file)

#Create a list of the bad indexes by only including the indexes present in all files
bad_freqs = []
bad_index = reduce(np.intersect1d,file_list)[:-1]

#Create a list of bad frequencies by using the bad indexes
for i in bad_index:
    bad_freqs = np.append(bad_freqs, freq_list[0][i])

print(bad_index)
#print(bad_freqs)

export_list_to_txt(bad_index, 'B0355+54_zapped_freqs.txt')
#export_list_to_txt(bad_freqs, 'bad_freqs_python.txt')
print('Array exported to file: B0355+54_zapped_freqs.txt')
