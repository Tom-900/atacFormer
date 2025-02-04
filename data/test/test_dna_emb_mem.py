import numpy as np
import psutil

dna_emb_table = np.random.randn(606200, 1280).astype(np.float32)
print(f"Memory occupied by the array: {dna_emb_table.nbytes / (1024 ** 2):.2f} MB")

# use memmap to save the memory
filename = '/lustre/project/Stat/s1155184322/datasets/atacFormer/dna_emb_table.dat'
dna_emb_table.tofile(filename)

# filename = '/lustre/project/Stat/s1155184322/datasets/atacFormer/dna_emb_table.npy'
# np.save(filename, dna_emb_table)

mmap_array = np.memmap(filename, dtype='float32', mode='r', shape=(606200, 1280))
print(f"Memory usage after creating memmap: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")

rows_to_read = np.random.randint(0, 600000, 5000)
selected_rows = mmap_array[rows_to_read]
print(f"Memory usage after accessing rows: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")

