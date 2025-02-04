import torch
from tqdm import tqdm
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from numba import njit, prange

# genome reference
genome_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/GRCh38.primary_assembly.genome.fa.h5"
genome = h5py.File(genome_file, 'r')

# full bin list
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/var_open_cells.txt"
bin_ls = []
with open(bin_file, "r") as f:
    for line in f:
        key, _ = line.strip().split('\t')
        bin_ls.append(key)
        

def extract_genome_data(genome):
    genome_data = {}
    for key in genome.keys():
        genome_data[key] = np.array(genome[key])
    return genome_data


@njit(parallel=True)
def extract_seq_numba(genome_data, chr, pos):
    seq_all = np.zeros((chr.shape[0], chr.shape[1], 5000))
    for i in prange(chr.shape[0]):
        for j in prange(chr.shape[1]):
            chr_ = str(chr[i, j]) if chr[i, j] != 23 else "X"
            chr_ = "chr" + chr_
            start, end = pos[i, j] * 5000, (pos[i, j] + 1) * 5000
            seq = genome_data[chr_][start:end]
            seq_all[i, j] = seq
            
    return seq_all


# TODO: expand the extraction to every data points in the batch
def extract_seq(genome_data, chr, pos):
    seq_all = np.zeros((chr.shape[0], chr.shape[1], 5000))
    for i in tqdm(range(chr.shape[0])):
        chr_, pos_ = chr[i], pos[i]
        for c in range(1, 24):
            chr_idx = np.where(chr_ == c)[0]
            pos_idx = pos_[chr_idx]
            pos_idx = np.array([np.arange(p * 5000, (p + 1) * 5000) for p in pos_idx], dtype=np.int32)
            chr_name = "chr" + str(c) if c < 23 else "chrX"
            
            seq = genome_data[chr_name][pos_idx]
            seq_all[i, chr_idx] = seq
            
    return seq_all


from joblib import Parallel, delayed
from multiprocessing import Array, Manager, shared_memory

def extract_seq_jb(genome_data, chr, pos):
    seq_all_shape = (chr.shape[0], chr.shape[1], 5000)
    shm = shared_memory.SharedMemory(create=True, size=np.prod(seq_all_shape) * np.dtype(np.float64).itemsize)
    seq_all = np.ndarray(seq_all_shape, dtype=np.float64, buffer=shm.buf)
    
    def process_chromosome(i):
        chr_, pos_ = chr[i], pos[i]
        for c in range(1, 24):
            chr_idx = np.where(chr_ == c)[0]
            pos_idx = pos_[chr_idx]
            pos_idx = np.array([np.arange(p * 5000, (p + 1) * 5000) for p in pos_idx], dtype=np.int32)
            chr_name = "chr" + str(c) if c < 23 else "chrX"
            
            seq = genome_data[chr_name][pos_idx]
            seq_all[i, chr_idx] = seq
    
    Parallel(n_jobs=-1)(delayed(process_chromosome)(i) for i in tqdm(range(chr.shape[0])))
    
    result = np.array(seq_all)
    shm.close()
    shm.unlink()
    
    return result


if __name__ == '__main__':
    chr = torch.randint(1, 24, (5, 20000), dtype=torch.long)
    pos = torch.randint(0, 8000, (5, 20000), dtype=torch.long)
    
    genome_data = extract_genome_data(genome)
    del genome
    # seq_all = extract_seq_numba(genome_data, chr.numpy(), pos.numpy())
    
    # seq_all = extract_seq(genome_data, chr.numpy(), pos.numpy())

    seq_all = extract_seq_jb(genome_data, chr.numpy(), pos.numpy())