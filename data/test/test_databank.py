import sys
sys.path.append("..")

from data.test.test_databank import DataBank
import scanpy as sc
from preprocess import preprocessor
from pathlib import Path


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
files = [f for f in output_dir.glob("*.h5ad")]

main_table_key = "X"
db = DataBank.from_anndata(
    adata_atac,
    to = output_dir / f"{files[0].stem.split('_')[0]}.scb",
    main_table_key=main_table_key,
    immediate_save=False,
)

db.meta_info.on_disk_format = "parquet"
db.sync()














