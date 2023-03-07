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
## Usage: dataset_prep.py -i <input file> -p <virus/host> -l <contig lenght> -m <minimum seq lenght> -n <max samples per file> 
##################################################

import os 
import sys
import pandas as pd # for EDA only
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import time
import optparse

startTime = time.time()   

# #### automated command line run
# prog_base = os.path.split(sys.argv[0])[1]
# parser = optparse.OptionParser()
# parser.add_option("-i", "--fileName", action = "store", type = "string", dest = "fileNameP",
# 									help = "Input name with path")
# parser.add_option("-p", "--contigType", action = "store", type = "string", dest = "contigType",
# 									help = "contigType, virus or host")
# parser.add_option("-l", "--contigLength", action = "store", type = int, dest = "contigLength",
# 									help = "lenght of the samples in bp")
# parser.add_option("-m", "--minSeqLen", action = "store", type = int, dest = "minSeqLen",
# 									help = "minimum lenght of the DNA", default = 500)
# parser.add_option("-n", "--samplesNumTr", action = "store", type = int, dest = "samplesNumTr",
# 									help = "number of samples for training", default = 1000000)

# (options, args) = parser.parse_args()
# if (options.fileNameP is None or options.contigLength is None or options.contigType is None) :
# 	sys.stderr.write(prog_base + ": ERROR: missing required command-line argument")
# 	parser.print_help()
# 	sys.exit(0)


# fileNameP = options.fileNameP
# filesDir = os.path.dirname(fileNameP)
# genomeFiles = [os.path.basename(fileNameP)]
# contigType =  options.contigType
# contigLength = options.contigLength
# contigLengthk = contigLength/1000
# if contigLengthk.is_integer() :
#     contigLengthk = int(contigLengthk)
# minSeqLen = options.minSeqLen
# samplesNumTr = options.samplesNumTr
# samplesNumVal = samplesNumTr * 0.22
# samplesNumTst = samplesNumTr * 0.22
# outputDir = './data_train'

#### manual run
startTime = time.time()   
contigType = 'virus'
contigLength = 150
minSeqLen = 500
contigLengthk = contigLength/1000
if contigLengthk.is_integer() :
    contigLengthk = int(contigLengthk)
filesDir = './datasets/Virome' 
genomeFiles = ['virus_db_10k.fa']
outputDir = './data_train'
samplesNumTr = 1000
samplesNumVal = 220
samplesNumTst = 220
outputDir = './data_train'

#############################################################
#### delete from raw file the sequences shorter than minSeqLen 

j = 0
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
print('Checking number of sequences too short...')
for fileName in genomeFiles:
    for seqRecord in SeqIO.parse(os.path.join(filesDir,fileName),"fasta"):
        # print(seqRecord.id)
        # print(repr(seqRecord.seq))
        # print(len(seqRecord))
        if len(seqRecord) < minSeqLen: #contigLength:
            # print(seqRecord.id)
            # print(repr(seqRecord.seq))
            # print(len(seqRecord))
            j += 1
            continue
        list1.append(seqRecord.id)
        list2.append(str(seqRecord.seq))
        list3.append(len(seqRecord))
        list4.append((str(seqRecord).count('N')))
print('Number of sequence elimanted too short: ', j)

df_cln = pd.DataFrame({'Seq_ID' : list1,
                       'Seq' : list2,
                            'Seq_Len': list3,
                            'Seq_NumN' : list4,
                           })
df_cln_sfl = df_cln.sample(frac=1).reset_index(drop=True)

#############################################################
#### remove all the contigs with more than 5% 'N'
print('Checking number of sequences too many N...')
j = 0
for row in df_cln_sfl.itertuples():
    NCBIName = row.Seq_ID
    seqTot = row.Seq
    pos = 0
    posEnd = (pos + contigLength)
    while posEnd <= len(seqTot) :
        seq = seqTot[pos:posEnd]
        contigName = ">"+NCBIName+"#"+str(contigLengthk)+"k#"+str(pos)+"#"+str(posEnd)+"\n"
        pos = posEnd + 1
        posEnd = pos + contigLength
        countN = seq.count("N")
        if countN/len(seq) <= 0.05:
            list5.append(contigName)
            list6.append(str(seq).upper())
        else:
            j += 1

print('Number of sequence elimanted too many N: ', j)


df_contigs = pd.DataFrame({'Seq_ID' : list5,
                            'Seq': list6,
                           })

#df_contigs2.head()
del list1, list2, list3, list4, list5, list6
del seqRecord, seqTot

#############################################################
#### generating contig of the desired lenght (contigLength)
print('splitting dataset')
X_train, X_test = train_test_split( df_contigs,
                                   train_size=0.70, 
                                   random_state=42,
                                   shuffle=False)  # set to false to avoid data leaks

X_val, X_test = train_test_split( X_test,
                                   train_size=0.50, 
                                   random_state=42,
                                   shuffle=False)  # set to false to avoid data leaks

del df_contigs

#############################################################
#### create training, validation and test files
print('saving files')
#############################################################
#### generation of the training file conmtaing samplesNumTr samples
print('saving training file')
datasetType = 'tr'
outFileName = contigType+'_'+datasetType+'_contigL_'+str(contigLengthk)+'k.fa'
filesDirOut = outputDir +'/'+datasetType+'/'
if not os.path.exists(filesDirOut):
    os.makedirs(filesDirOut)
output1 = open(os.path.join(filesDirOut, outFileName), "w" )
X_train.reset_index()
i = 0
for row in X_train.itertuples():
    contigName = row.Seq_ID
    L = [contigName, row.Seq +"\n"]
    output1.writelines(L) 
output1.close()
del X_train

### check samples num output
print('number of samples training file')
cmd2 = 'grep ">" '+filesDirOut+outFileName+' | wc -l'
os.system(cmd2)
seqNumSmpld = os.system(cmd2)

#### subsample 'samplesNumTr' reads 
samplesNumTrk = samplesNumTr/1000
if samplesNumTrk.is_integer() :
    samplesNumTrk = int(samplesNumTrk)
inFile = os.path.join(filesDirOut,outFileName)
outFile2 = contigType+'_'+datasetType+'_smp_'+str(samplesNumTrk)+'k.fa'
outFile = os.path.join(filesDirOut,outFile2)
cmd3 = 'reformat.sh in='+inFile+' out='+outFile+' samplereadstarget='+str(samplesNumTr)+' sampleseed=42 ignorejunk'
os.system(cmd3)

#############################################################
#### generation of the validation file conmtaing samplesNumVal samples
print('saving validation file')
datasetType = 'val'
outFileName = contigType+'_'+datasetType+'_contigL_'+str(contigLengthk)+'k.fa'
filesDirOut = outputDir +'/'+datasetType+'/'
if not os.path.exists(filesDirOut):
    os.makedirs(filesDirOut)
output1 = open(os.path.join(filesDirOut, outFileName), "w" )
X_val.reset_index()
i = 0
for row in X_val.itertuples():
    contigName = row.Seq_ID
    L = [contigName, row.Seq +"\n"]
    output1.writelines(L) 
output1.close()
del X_val

### check samples num output
print('number of samples validation file')
cmd2 = 'grep ">" '+filesDirOut+outFileName+' | wc -l'
os.system(cmd2)
seqNumSmpld = os.system(cmd2)

#### subsample 'samplesNumVal' reads 
samplesNumValk = samplesNumVal/1000
if samplesNumValk.is_integer() :
    samplesNumValk = int(samplesNumValk)
inFile = os.path.join(filesDirOut,outFileName)
outFile2 = contigType+'_'+datasetType+'_smp_'+str(samplesNumValk)+'k.fa'
outFile = os.path.join(filesDirOut,outFile2)
cmd3 = 'reformat.sh in='+inFile+' out='+outFile+' samplereadstarget='+str(samplesNumVal)+' sampleseed=42 ignorejunk'
os.system(cmd3)

#############################################################
#### generation of the validation file conmtaing samplesNumVal samples
print('saving test file')
datasetType = 'tst'
outFileName = contigType+'_'+datasetType+'_contigL_'+str(contigLengthk)+'k.fa'
filesDirOut = outputDir +'/'+datasetType+'/'
if not os.path.exists(filesDirOut):
    os.makedirs(filesDirOut)
output1 = open(os.path.join(filesDirOut, outFileName), "w" )
X_test.reset_index()
i = 0
for row in X_test.itertuples():
    contigName = row.Seq_ID
    L = [contigName, row.Seq +"\n"]
    output1.writelines(L) 
output1.close()
del X_test

### check samples num output
print('number of samples test file')
cmd2 = 'grep ">" '+filesDirOut+outFileName+' | wc -l'
os.system(cmd2)
seqNumSmpld = os.system(cmd2)

#### subsample 'samplesNumTst' reads 
samplesNumTstk = samplesNumTst/1000
if samplesNumTstk.is_integer() :
    samplesNumTstk = int(samplesNumTstk)
inFile = os.path.join(filesDirOut,outFileName)
outFile2 = contigType+'_'+datasetType+'_smp_'+str(samplesNumTstk)+'k.fa'
outFile = os.path.join(filesDirOut,outFile2)
cmd3 = 'reformat.sh in='+inFile+' out='+outFile+' samplereadstarget='+str(samplesNumTst)+' sampleseed=42 ignorejunk'
os.system(cmd3)

endTime = time.time() - startTime         
print('Execution time', endTime/60)