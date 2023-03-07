#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################################
## Predicting sequences with the trained model
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
## Usage: VirExplore.py -i <input> -l <contig lenght> - n <max samples per file> -m <model dir> -o <output dir> -c <cutoff lenght> 
##################################################

import os
import time
import csv
import sys
import optparse
import numpy as np
from Bio import SeqIO
from tensorflow.keras.models import load_model
from more_itertools import chunked

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))

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
                entry = next(iterator)  # iterator.next()
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch


start_time = time.time()

# #### Automated run
prog_base = os.path.split(sys.argv[0])[1]
parser = optparse.OptionParser()
parser.add_option("-i", "--in", action = "store", type = "string", dest = "fileNameP", 
                  help = "input file in fasta format with path")
parser.add_option("-l", "--len", action = "store", type = int, dest = "contigLength",
 									help = "contig Length")
parser.add_option("-n", "--samplesNum", action = "store", type = int, dest = "samplesNum",
									help = "number of sequences for each split file", default = 100000)
parser.add_option("-m", "--mod", action = "store", type = "string", dest = "modDir",
 									default='./data_train/models', 
                  help = "model directory (default ./data_train_models)")
parser.add_option("-o", "--out", action = "store", type = "string", dest = "outDir",
 									default='./output', help = "output directory (default ./output")
parser.add_option("-c", "--ctf", action = "store", type = "int", dest = "cutoffLen",
 									default=80, help = "predict only for sequence >= l bp (default 80)")


(options, args) = parser.parse_args()
if (options.fileNameP is None or  options.contigLength is None):
 	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
#	filelog.write(prog_base + ": ERROR: missing required command-line argument")
 	parser.print_help()
 	sys.exit(0)
    
fileNameP = options.fileNameP
fileName = os.path.basename(fileNameP)
fileDir = os.path.dirname(fileNameP)
numSeqsxFile = options.samplesNum
modDir = options.modDir
outDir = options.outDir
if not os.path.exists(outDir):
    os.makedirs(outDir)
cutoffLen = options.cutoffLen
contigLength = options.contigLength
contigLengthk = str(contigLength/1000)

#### Manula run

# fileNameP = './data_test/test_20.fa'
# fileName = os.path.basename(fileNameP)
# fileDir = os.path.dirname(fileNameP)
# outDir = './output'
# modDir = './data_train/models'

# if not os.path.exists(outDir):
#     os.makedirs(outDir)

# cutoff_len = 80
# contigLength = 100
# contigLengthk = str(contigLength/1000)
# numSeqsxFile = 10000 # number of sequence for each temporary file

#### Step 1: load model ####
print("1. Loading Models.")
print("   model directory {}".format(modDir))

## loading model and creating null dictionary for p-value evaluation
modDict = {}
nullDict = {}
modPattern = 'model_CNN_5Layers_' + contigLengthk + 'k'
modName = [x for x in os.listdir(modDir) if modPattern in x and x.endswith(".h5")][0]

load_model(os.path.join(modDir, modName))
modDict[contigLengthk] = load_model(os.path.join(modDir, modName))

Y_pred_file = [x for x in os.listdir(modDir) if modPattern in x and "Y_pred" in x][0]

with open(os.path.join(modDir, Y_pred_file)) as f:
    tmp = [line.split() for line in f][0]
    Y_pred = [float(x) for x in tmp]
Y_true_file = [x for x in os.listdir(modDir) if modPattern in x and "Y_true" in x][0]

with open(os.path.join(modDir, Y_true_file)) as f:
    tmp = [line.split()[0] for line in f]
    Y_true = [float(x) for x in tmp]
nullDict[contigLengthk] = Y_pred[:Y_true.index(1)]

model = modDict[contigLengthk]
null = nullDict[contigLengthk]

end_time1 = time.time() - start_time
print('Execution time prepaing files', end_time1 / 60)

print("2. Encoding Sequences.")

### check samples num output
cmd2 = 'grep ">" ' + fileNameP + ' | wc -l'
seqNumSmpld = os.system(cmd2)


# NCBIName = os.path.splitext((os.path.basename(fileName)))[0]
# fileDir = os.path.dirname(fileName)

#contigLength = 150
#contigLengthk = contigLength / 1000
# if contigLengthk.is_integer():
#     contigLengthk = int(contigLengthk)

outDir0 = fileDir
outDir1 = os.path.join(outDir0, "encode")
outDir2 = os.path.join(outDir1, "temp")

if not os.path.exists(outDir1):
    os.makedirs(outDir1)
if not os.path.exists(outDir2):
    os.makedirs(outDir2)

# splitting input file in smaller files containing numSeqsxFile sequences
record_iter = SeqIO.parse(open(fileNameP), "fasta")
for i, batch in enumerate(batch_iterator(record_iter, numSeqsxFile)):
    filename = "split_%i_.fa_t" % (100 + i)
    with open(os.path.join(outDir2, filename), "w") as handle:
        count = SeqIO.write(batch, handle, "fasta")
    print("Wrote %i records to %s" % (count, filename))

fileList = [f for f in os.listdir(outDir2) if f.endswith('.fa_t')]
fileList.sort()
num_tot_seq = 0
j = 0
fileCount = 1
list_lftovr1 = []
list_lftovr2 = []

for file in fileList:
    print('Processing file: ', file)
    fileNameT = os.path.splitext((os.path.basename(file)))[0]
    codeFileNamefw = fileNameT+"#"+str(contigLengthk)+"k_"+"seq_"+"num-"+str(100+fileCount)+"_codefw.npy"    
    nameFileName = fileNameT+"#"+str(contigLengthk)+"k_num"+str(100+fileCount)+"_seq.fasta" 
    with open(os.path.join(outDir1, codeFileNamefw), "bw") as f:
        
        records = SeqIO.parse(os.path.join(outDir2,file),"fasta")
        for batch in chunked(records, numSeqsxFile):
            list_seq_coded = []
            seqname = []
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
                
                seqname.append(">"+seqRecord.id)
                seqname.append(str(seqRecord.seq))
                
            
            np.save( f, np.array(list_seq_coded ,dtype = np.ubyte))
            seqnameF = open(os.path.join(outDir1, nameFileName), "a")
            seqnameF.write('\n'.join(seqname) + '\n')
            seqnameF.close()

            
            num_tot_seq = num_tot_seq+ len(list_seq_coded)
        fileCount +=1     
        
    print("Encoded sequences are saved in {}".format(codeFileNamefw))
    print('Total number sequences deleted: {}'.format(j))   
    print('Total number sequences encoded: {}'.format(num_tot_seq))


end_time2 = time.time() - end_time1
print('Execution time encoding', end_time2 / 60)

print("3. Predicting Sequences.")
filenames_code = [x for x in os.listdir(outDir1) if 'codefw.npy' in x and str(contigLengthk) + 'k' in x]
filenames_code.sort()
filenames_seq = [x for x in os.listdir(outDir1) if 'seq.fasta' in x and str(contigLengthk) + 'k' in x]
filenames_seq.sort()
# clean the output file
outfile = os.path.join(outDir, os.path.basename(fileName) + '_gt' + str(cutoffLen) + 'bp_dvfpred.txt')
predF = open(outfile, 'w')
writef = predF.write('\t'.join(['name', 'len', 'score', 'pvalue']) + '\n')
predF.close()
predF = open(outfile, 'a')

### generating outpu file with all predicitions scores
for fname_code, fname_seq in zip(filenames_code, filenames_seq):
    temp = np.array([np.load(os.path.join(outDir1, fname_code))], dtype=np.float16)[0]
    score = model.predict(temp)
    score1 = score[:, 0]
    pvalue = sum([x > score for x in null]) / len(null)
    pvalue1 = pvalue[:, 0]
    head = []
    seqL = []
    for seqRecord in SeqIO.parse(os.path.join(outDir1, fname_seq), "fasta"):
        # print(seqRecord.id)
        # print(repr(seqRecord.seq))
        # print(len(seqRecord))
        head.append(seqRecord.id)
        seqL.append(len(seqRecord))

    with open(outfile, "a+") as file:
        writer = csv.writer(file, delimiter='\t')
        for row in zip(head, seqL, score1, pvalue1):
            writer.writerow(row)
    endTime = time.time() - start_time
    print('Execution time writing file', round(endTime/60,2))

predF.close()

end_time3 = time.time() - end_time2
print('Execution time predicitng', round(end_time3/60,2))
end_time_tot = time.time() - start_time
print('Total execution time', round(end_time_tot/60,2))


