import sys
sys.path.append("..")

from databank import DataBank
import scanpy as sc
from preprocess import preprocessor
from pathlib import Path
import numpy as np

atac_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/HBM233.GKRM.627_ATAC.h5ad"
bin_file = "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt"
adata_atac = sc.read(atac_file)

adata_atac = preprocessor(
            adata_atac, 
            filter_bin=True,
            intersect=True, # for open bins only
            bin_file=bin_file,
            filter_cell_by_bins=100,
            batch_size=10000,
        )

output_dir = Path("/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart")

main_table_key = "X"
db = DataBank.from_anndata(
    adata_atac,
    to = output_dir / "HBM233.GKRM.627.scb",
    main_table_key=main_table_key,
    immediate_save=False,
)

# check the data
db.data_tables["X"].data[0]["chr_id"]
db.data_tables["X"].data[0]["pos_id"]

adata_ls = adata_atac.var_names[np.nonzero(adata_atac[0].X.A)[1]].tolist()
chr_id = [23 if i.split(":")[0]=="X" else int(i.split(":")[0]) for i in adata_ls]
pos_id = [int((int(bin_name.split(":")[1].split("-")[0]) - 1) / 5000 + 1) for bin_name in adata_ls]

print(chr_id == db.data_tables["X"].data[0]["chr_id"])
print(pos_id == db.data_tables["X"].data[0]["pos_id"])

# save the databank object
db.meta_info.on_disk_format = "parquet"
db.sync()














