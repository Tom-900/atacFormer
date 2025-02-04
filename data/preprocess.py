from typing import Optional, Union
import numpy as np
from scipy.sparse import issparse, csr_matrix, SparseEfficiencyWarning
import scanpy as sc
from anndata import AnnData
from pathlib import Path
from tqdm import tqdm
import warnings
import pandas as pd


def preprocessor(
    adata: AnnData,
    filter_bin:bool = False,
    intersect: bool = True,
    bin_file: Optional[str] = None,
    filter_cell_by_bins: Union[int, bool] = False,
    batch_size: int = 100000,
    ):
    r"""
    Args:
    filter_bin (:class:`bool`, default: ``False``):
        Whether to filter bins, if True, filter bins that in the bin_file
    intersect (:class:`bool`, default: ``False``): 
        Whether to intersect the bins in the bin file with the bins in the adata object
    bin_file (:class:`str`, optional):
        The file path to save the (filtered) bins
    filter_cell_by_bins (:class:`int` or :class:`bool`, default: ``False``):
        Whether to filter cells by No. of open bins, if :class:`int`, filter cells with counts
    batch_size (:class:`int`, default: ``10000``):
        The batch size of bins to process the data
    """
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        
    # step 1: filter bins
    if filter_bin:
        print("Filtering bins ...")
        assert bin_file is not None, "Please provide the bin file path"

        bin_table = pd.read_table(bin_file, header=None)
        bin_ls = bin_table.iloc[:, 0].tolist()
        bin_set = set(bin_ls)
            
        if not intersect:
            # Create a new AnnData object with the filtered bins
            # bdata = AnnData(csr_matrix((adata.n_obs, len(bin_set))), var={"var_names": list(bin_set)})
            bdata = AnnData(csr_matrix((adata.n_obs, len(bin_ls))), var={"var_names": bin_ls})
            
            for key in adata.obs.keys():
                bdata.obs[key] = adata.obs[key].copy()
            
            for key in adata.obsm.keys():
                bdata.obsm[key] = adata.obsm[key].copy()
            
            # Fill in the values for the bins that are in both bin_set and adata
            common_bins = [bin_name for bin_name in adata.var_names if bin_name in bin_set]
            bins_in_adata = [adata.var_names.get_loc(bin_name) for bin_name in common_bins if bin_name in adata.var_names]
            bins_in_bdata = [bdata.var_names.get_loc(bin_name) for bin_name in common_bins if bin_name in bdata.var_names]
        
            adata_csc = adata.X.tocsc()
            bdata_csc = bdata.X.tocsc()
        
            while batch_size > 0:
                print(f"Current batch size: {batch_size}")
                try:
                    for i in tqdm(range((len(bins_in_adata) // batch_size) + 1)):
                        bdata_csc[:, bins_in_bdata[i*batch_size :min((i+1)*batch_size, len(bins_in_bdata))]] = \
                            adata_csc[:, bins_in_adata[i*batch_size: min((i+1)*batch_size, len(bins_in_adata))]]
                    bdata.X = bdata_csc.tocsr()
                    break
                except:
                    print("Batch size too large, reducing batch size ...")
                    batch_size = batch_size // 10 
                    continue
        else:
            common_bins = [bin_name for bin_name in adata.var_names if bin_name in bin_set]
            adata = adata[:, common_bins]
            
            bdata = adata.copy()
    else:
        bdata = adata.copy()
            
    # step 2: filter cells
    if isinstance(filter_cell_by_bins, int) and filter_cell_by_bins > 0:
        print("Filtering cells by counts ...")
        sc.pp.filter_cells(bdata, min_genes=filter_cell_by_bins if isinstance(filter_cell_by_bins, int) else None)

    # step 3: binarization
    print("Binarizing data ...")
    bdata.X = (bdata.X > 0)
        
    return bdata
    
    
# testing the function
if __name__ == '__main__':

    data_dir = Path("/lustre/project/Stat/s1155184322/datasets/atacGPT/HuBMAP/heart")
    adata_atac = sc.read_h5ad(data_dir / "HBM233.GKRM.627_ATAC.h5ad")
    print(adata_atac)
    
    filtered_bins_file = "/lustre/project/Stat/s1155184322/datasets/atacGPT/HuBMAP/var_open_cells_filtered.txt"
    adata_atac = preprocessor(adata_atac, filter_bin=True, bin_file=filtered_bins_file, filter_cell_by_bins=1000)
    print(adata_atac)