import pandas as pd
import numpy as np
import processSeq
import sys


PATH1='.' # director of dataset

def gen_Seq(Range):
	print ("Generating Seq...")	
	table = pd.read_table(PATH1+"prep_data.txt",sep = "\t")
	print (len(table))
	table.drop_duplicates()
	print (len(table))
	label_file = open(PATH1+"LabelSeq", "w")

	total = len(table)

	list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
			"chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
			"chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY","chrM"]

	number_positive = 0
	dict_pos={}
	genome_assemblyPath = PATH1+"Chromosome_38/"

	for i in range(total):
		
		if (number_positive % 100 == 0) and (number_positive != 0):
			print ("number of seq: %d of %d\r" %(number_positive,total),end = "")
			sys.stdout.flush()

		chromosome = table["chromosome"][i]
		if chromosome in dict_pos.keys():
			strs = dict_pos[chromosome]
		else:
			strs = processSeq.getString(genome_assemblyPath + str(chromosome) + ".fa")
			dict_pos[chromosome] = strs

		bias = 7
		start = int(table["start"][i] - 1 - Range + bias)
		end = start + 23 + Range*2
		
		strand = table["strand"][i]
		
		edstrs1 = strs[start : end]

		if strand == "-":
			edstrs1 = edstrs1[::-1]
			edstrs1 = processSeq.get_reverse_str(edstrs1)
		
		if "N" in edstrs1:
			table = table.drop(i)
			continue

		outstr = "%s\n"%(edstrs1)
		label_file.write(outstr)
		number_positive += 1
	table.to_csv(PATH1+"prep_data.txt",sep = "\t",index = False)

def get_target():
	table = pd.read_table(PATH1+"prep_data.txt", sep="\t")
	print (len(table))
	table.drop_duplicates()
	print (len(table))
	target_file = open(PATH1+"TargetSeq", "w")
	for i in range(len(table)):
		target = table['target'][i].upper()
		target_file.write(target+"\n")
	target_file.close()

def prep_data():
	chrom_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", \
		"chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", \
		"chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY","chrM"]
	tab = pd.read_table(PATH1+"casoffinder_CHANGEseq_joined.tsv",sep = '\t')
	tab  = tab[tab['chromosome'].isin(chrom_list)]
	tab['label'] = 1 - tab['reads'].isna()
	tab['end'] = tab['start'] + 23
	print (tab['chromosome'].unique())

	tab.to_csv(PATH1+"prep_data.txt",sep = "\t",index = False)

def load_file(f_name,length,vec_name):
	base_code = {
			'A': 0,
			'C': 1,
			'G': 2,
			'T': 3,
		}
	num_pairs = sum(1 for line in open(f_name))
	# number of sample pairs
	num_bases = 4

	with open(f_name, 'r') as f:
		line_num = 0 # number of lines (i.e., samples) read so far
		for line in f.read().splitlines():
			if (line_num % 100 == 0) and (line_num != 0):
				print ("number of input data: %d\r" %(line_num),end= "")
				sys.stdout.flush()

			if line_num == 0:
				# allocate space for output
				seg_length = length # number of bases per sample
				Xs_seq1 = np.zeros((num_pairs, num_bases, seg_length))
				

				for start in range(len(line)):
					if line[start] in base_code:
						print (start)
						break

			base_num = 0
			
			for x in line[start:start+length]:
				if x != "N":
					Xs_seq1[line_num, base_code[x], base_num] = 1
				base_num += 1
			line_num += 1
	X = Xs_seq1
	np.save("../%s" %(vec_name),X)
	

prep_data()
gen_Seq(100)
load_file(PATH1+"/LabelSeq",223,"vec.npy")
get_target()
load_file(PATH1+"/TargetSeq",23,"t.npy")

