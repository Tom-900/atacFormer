import numpy as np

dna_emb = np.zeros((606224, 1280))
np.save('/lustre/project/Stat/s1155184322/datasets/atacFormer/dna_emb_table.npy', dna_emb)