import pandas as pd
import h5py

# load all the bin name (chromosome, start, end)
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/var_open_cells_23chr.txt"
bin_table = pd.read_table(bin_file, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()

# load human genome reference
# this file can be downloaded from https://www.synapse.org/Synapse:syn52559388/files/
genome_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/GRCh38.primary_assembly.genome.fa.h5"
genome = h5py.File(genome_file, 'r')

# one example
bin_name = bin_ls[0]
chr, start, end = "chr" + bin_name.split(":")[0], int(bin_name.split(":")[1].split("-")[0]), int(bin_name.split(":")[1].split("-")[1])

# 1-A, 2-C, 3-G, 4-T, 0-N (unknown)
seq = genome[chr][start - 1: end - 1]

# extract the DNA sequence embedding by Nucleotide Transformer
# tutorial: https://github.com/instadeepai/nucleotide-transformer/blob/main/examples/inference.ipynb
# do this for all the bins in the bin_ls

# save the result into a numpy array with shape (len(bin_ls) + 1, emb_dim) = (606199 + 1, 1280)
# the first row is for padding, which is all zeros
# 1280 can be replaced by other embedding dim in different pretrained models


