#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################################
## One-hot encoding for the files used for training
##################################################
## {License_info}
##################################################
## Author: Vittorio Pepe
## Copyright: Copyright 2021, VirExplorer
## Credits: DeepVirFinder
## License: {license}
## Version: 1.0.0
## Mmaintainer: Vittorio Pepe
## Email: vpepe.ds@gmail.com
## Status: development
## Usage: encode.py -i <input> -l <contig lenght> - n <max samples per file> 
##################################################

import os
import sys
import time
import shutil
import optparse
import numpy as np
from Bio import SeqIO
from more_itertools import chunked

def encodeSeq5(seq) : 
    seq_code = list()
    for pos in range(len(seq)) :
        letter = seq[pos]
        if letter in ['A', 'a'] :
            code = [1,0,0,0,0]
        elif letter in ['C', 'c'] :
            code = [0,1,0,0,0]
        elif letter in ['G', 'g'] :
            code = [0,0,1,0,0]
        elif letter in ['T', 't'] :
            code = [0,0,0,1,0]
        else :
            code = [0,0,0,0,1]
        seq_code.append(code)
    return seq_code 

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)   #iterator.next()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch

startTime = time.time()    

#### automated command line run
prog_base = os.path.split(sys.argv[0])[1]
parser = optparse.OptionParser()
parser.add_option("-i", "--fileName", action = "store", type = "string", dest = "fileNameP",
									help = "Input name with path")
parser.add_option("-l", "--contigLength", action = "store", type = int, dest = "contigLength",
									help = "lenght of the samples in bp")
parser.add_option("-n", "--samplesNum", action = "store", type = int, dest = "samplesNum",
									help = "number of sequences for each split file", default = 500000)

(options, args) = parser.parse_args()
if (options.fileNameP is None or options.contigLength is None):
	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
	parser.print_help()
	sys.exit(0)

fileNameP = options.fileNameP
fileDir = os.path.dirname(fileNameP)
fileName = os.path.basename(fileNameP)
fileNameP = options.fileNameP
contigType = fileName.split('_')[0]
contigLength = options.contigLength
contigLengthk = contigLength/1000
numSeqsxFile = options.samplesNum

#### manual run
# fileNameP = './data_train/tr/host_tr_smp_1000k.fa'
# fileName = os.path.basename(fileNameP)
# contigType = fileName.split('_')[0] #'host' #options.contigType
# contigLength = 100 # options.contigLength
# #chunks = 500000
# numSeqsxFile = 500000
# fileDir = os.path.dirname(fileName)

#############################################################
#### generation of encoded split binary files to limit RAM usage
list_in_fa_names = [fileNameP]

list1 = []
list2 = []
list3 = []
list_lftovr1 = []
list_lftovr2 = []
list6 = []

for fileName in list_in_fa_names:    
    startTime = time.time()  
    os.getcwd()
    ### check samples num output
    cmd2 = 'grep ">" '+fileName+' | wc -l'
    print('Number of total sequences to be encoded: ')
    os.system(cmd2)
    NCBIName = os.path.splitext((os.path.basename(fileName)))[0]
    fileDir = os.path.dirname(fileName)
    outDir0 = fileDir
    outDir = os.path.join(outDir0, "encode")
    outDir2 =  os.path.join(outDir, "temp")
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    if not os.path.exists(outDir2):
        os.makedirs(outDir2)
    else:
        shutil.rmtree(outDir2)
        os.makedirs(outDir2)
        
    record_iter = SeqIO.parse(open(fileName), "fasta")
    for i, batch in enumerate(batch_iterator(record_iter, numSeqsxFile)):
        filename = "split_" + contigType + "_%i_.fa_t" % (101 + i)
        with open(os.path.join(outDir2, filename), "w") as handle:       
            count = SeqIO.write(batch, handle, "fasta")
        print("Wrote %i records to %s" % (count, filename))

    fileList =  [f for f in os.listdir(outDir2) if f.endswith('.fa_t')]
    fileList.sort()
    num_tot_seq = 0
    j = 0
    fileCount = 1       
    for file in fileList:
        print('Processing file: ', file)
        NCBIName = os.path.splitext((os.path.basename(file)))[0]
        codeFileNamefw = contigType+"#"+NCBIName+"#"+str(contigLengthk)+"k_"+"seq_"+"num-"+str(100+fileCount)+"_codefw.npy"

        with open(os.path.join(outDir, codeFileNamefw), "bw") as f:
            records = SeqIO.parse(os.path.join(outDir2,file),"fasta")
            for batch in chunked(records, numSeqsxFile):   # replaced chunks with numSeqsxFile
                list_seq_coded = []
                for seqRecord in batch:
                    # print(seqRecord.id)
                    # print(repr(seqRecord.seq))
                    # print(len(seqRecord))
                    if len(seqRecord) < contigLength:
                        print('Too short sequence!')
                        print(seqRecord.id)
                        print(repr(seqRecord.seq))
                        print(len(seqRecord))
                        list_lftovr1.append(seqRecord.id)
                        list_lftovr2.append(seqRecord.seq)
                        j += 1
                        continue
                    countN = str(seqRecord.seq).count("N")
                    if countN/len(seqRecord.seq) >= 0.3:
                        print('More than 30% N in sequence!')
                        print(seqRecord.id)
                        print(repr(seqRecord.seq))
                        print(len(seqRecord))
                        list_lftovr1.append(seqRecord.id)
                        list_lftovr2.append(seqRecord.seq)
                        j += 1
                        continue
                    
                    seq_coded = encodeSeq5(seqRecord.seq)
                    list_seq_coded.append(seq_coded)
                
                np.save(f, np.array(list_seq_coded ,dtype = np.ubyte))
                num_tot_seq = num_tot_seq+ len(list_seq_coded)
            fileCount +=1     
            
        print("Encoded sequences are saved in {}".format(codeFileNamefw))
        print('Total number sequences deleted: {}'.format(j))   
        print('Total number sequences encoded: {}'.format(num_tot_seq))
        
shutil.rmtree(outDir2)
endTime = time.time() - startTime         
print('Execution time', round(endTime/60,2))
