# modified from https://github.com/bowang-lab/scGPT/tree/main/scgpt/scbank
import json
from pathlib import Path
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal

import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from anndata import AnnData
from datasets import Dataset, load_dataset
from scvi import settings

import logging
import sys

# set up logger
logger = logging.getLogger("DataBank")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class DataTable:
    """
    The data structure for a single-cell data table.
    """

    name: str
    data: Optional[Dataset] = None

    @property
    def is_loaded(self) -> bool:
        return self.data is not None and isinstance(self.data, Dataset)

    def save(
        self,
        path: Union[Path, str],
        format: Literal["json", "parquet"] = "json",
    ) -> None:
        if not self.is_loaded:
            raise ValueError("DataTable is not loaded.")

        if isinstance(path, str):
            path = Path(path)

        if format == "json":
            self.data.to_json(path)
        elif format == "parquet":
            self.data.to_parquet(path)
        else:
            raise ValueError(f"Unknown format: {format}")


@dataclass
class MetaInfo:
    """
    The data structure for meta info of a scBank data directory.
    """

    on_disk_path: Union[Path, str, None] = None
    on_disk_format: Literal["json", "parquet"] = "json"
    main_table_key: Optional[str] = None

    def __post_init__(self):
        if self.on_disk_path is not None:
            self.on_disk_path: Path = Path(self.on_disk_path)

    def save(self, 
             path: Union[Path, str, None] = None,
             suffix: Optional[str] = None) -> None:
        """
        Save meta info to path. If path is None, will save to the same path at
        :attr:`on_disk_path`.
        """
        if path is None:
            path = self.on_disk_path
        if isinstance(path, str):
            path = Path(path)

        manifests = {
            "on_disk_format": self.on_disk_format,
            "main_data": self.main_table_key,
        }
        if suffix is not None and isinstance(suffix, str):
            with open(path / f"manifest.{suffix}.json", "w") as f:
                json.dump(manifests, f, indent=2)
        else:
            with open(path / f"manifest.json", "w") as f:
                json.dump(manifests, f, indent=2)

    def load(self, path: Union[Path, str, None] = None) -> None:
        """
        Load meta info from path. If path is None, will load from the same path
        at :attr:`on_disk_path`.
        """
        if path is None:
            path = self.on_disk_path
        if isinstance(path, str):
            path = Path(path)

        with open(path / "manifest.json") as f:
            manifests = json.load(f)
        self.on_disk_format = manifests["on_disk_format"]
        self.main_table_key = manifests["main_data"]

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> Self:
        """
        Create a MetaInfo object from a path.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")
        if not (path / "manifest.json").exists():
            raise ValueError(f"Path {path} does not contain manifest.json.")

        meta_info = cls()
        meta_info.on_disk_path = path
        meta_info.load(path)
        return meta_info
    
    
@dataclass
class Setting:
    """
    The configuration for scBank :class:`DataBank`.
    """

    remove_zero_rows: bool = field(
        default=True,
        metadata={
            "help": "When load data from numpy or sparse matrix, "
            "whether to remove rows with zero values."
        },
    )
    max_tokenize_batch_size: int = field(
        default=1e6,
        metadata={
            "help": "Maximum number of cells to tokenize in a batch. "
            "May be useful for processing numpy arrays, currently not used."
        },
    )
    immediate_save: bool = field(
        default=False,
        metadata={
            "help": "Whether to save DataBank whenever it is initiated or updated."
        },
    )
    
    
@dataclass
class DataBank:

    meta_info: MetaInfo = None
    data_tables: Dict[str, DataTable] = field(
        default_factory=dict,
        metadata={"help": "Data tables in the DataBank."},
    )
    settings: Setting = field(
        default_factory=Setting,
        metadata={"help": "The settings for scBank, use default if not set."},
    )

    def __post_init__(self) -> None:
        """
        Initialize a DataBank. If initialize a non-empty DataBank, will check the
        meta info and import the data tables. The main data table indicator will be
        required, if at least one data table is provided.
        """

        # empty initialization:
        if self.meta_info is None:
            if len(self.data_tables) > 0:
                raise ValueError("Need to provide meta info if non-empty data tables.")
            logger.debug("Initialize an empty DataBank.")
            return

        # only-meta-info initializtion:
        if len(self.data_tables) == 0:
            logger.debug("DataBank initialized with meta info only.")
            if self.settings.immediate_save:
                self.sync()
            return

        # validate the meta info, and the consistency between meta and data tables
        # we assume the input meta_info is complete and correct. Usually this should
        # be handled by factory constructors.
        self._validate_data()
        if self.settings.immediate_save:
            self.sync()

    @property
    def main_table_key(self) -> Optional[str]:
        """
        The main data table key.
        """
        if self.meta_info is None:
            return None
        return self.meta_info.main_table_key

    @main_table_key.setter
    def main_table_key(self, table_key: str) -> None:
        """Set the main data table key."""
        if self.meta_info is None:
            raise ValueError("Need to have self.meta_info if setting main table key.")
        self.meta_info.main_table_key = table_key
        if self.settings.immediate_save:
            self.sync(["meta_info"])

    @property
    def main_data(self) -> DataTable:
        """The main data table."""
        return self.data_tables[self.main_table_key]

    @classmethod
    def from_anndata(
        cls,
        adata: Union[AnnData, Path, str],
        to: Union[Path, str],
        main_table_key: str = "X",
        token_col: Optional[str] = None,
        immediate_save: bool = True,
    ) -> Self:
        """
        Create a DataBank from an AnnData object.

        Args:
            adata (AnnData): Annotated data or path to anndata file.
            to (Path or str): Data directory.
            main_table_key (str): This layer/obsm in anndata will be used as the
                main data table.
            token_col (str): Column name of the gene token.
            immediate_save (bool): Whether to save the data immediately after creation.
        Returns:
            DataBank: DataBank instance.
        """

        if isinstance(adata, str) or isinstance(adata, Path):
            import scanpy as sc
            adata = sc.read(adata, cache=True)
        elif not isinstance(adata, AnnData):
            raise ValueError("adata must points to an AnnData object.")

        if isinstance(to, str):
            to = Path(to)
            
        to.mkdir(parents=True, exist_ok=True)
        db = cls(
            meta_info=MetaInfo(on_disk_path=to),
            settings=Setting(immediate_save=immediate_save),
        )

        # TODO: Add other data tables, currently only read the main data
        data_table = db.load_anndata(
            adata,
            data_keys=[main_table_key],
            token_col=token_col,
        )[0]
        
        # update and immediate save
        db.main_table_key = main_table_key
        db.update_datatables(new_tables=[data_table], immediate_save=immediate_save)

        return db
    
    def load_anndata(
        self,
        adata: AnnData,
        data_keys: Optional[List[str]] = None,
        token_col: str = "gene name",
    ) -> List[DataTable]:
        """
        Load anndata into datatables.

        Args:
            adata (:class:`AnnData`): Annotated data object to load.
            data_keys (list of :obj:`str`): List of data keys to load. If None,
                all data keys in :attr:`adata.X` and :attr:`adata.layers` will be
                loaded.
            token_col (:obj:`str`): Column name of the gene token. Tokens will be
                converted to indices by :attr:`self.gene_vocab`.
        Returns:
            list of :class:`DataTable`: List of data tables loaded.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("adata must be an AnnData object.")

        if data_keys is None:
            data_keys = ["X"] + list(adata.layers.keys())

        if token_col is not None:
            if token_col not in adata.var:
                raise ValueError(f"token_col {token_col} not found in adata.var.")
            if not isinstance(adata.var[token_col][0], str):
                raise ValueError(f"token_col {token_col} must be of type str.")

        # validate matching between tokens and vocab
        if token_col is not None:
            tokens = adata.var[token_col].tolist()
        else:
            tokens = adata.var_names.tolist()
            
        # buld chr, chr_pos

        # build mapping to scBank datatable keys
        # _ind2ind = _map_ind(tokens, self.atac_vocab)  # old index to new index
        chr = [23 if i.split(":")[0]=="X" else int(i.split(":")[0]) for i in tokens]
        pos = [int((int(bin_name.split(":")[1].split("-")[0]) - 1) / 5000 + 1) for bin_name in tokens]

        data_tables = []
        for key in data_keys:
            data = self._load_anndata_layer(adata, chr, pos, key)
            data_table = DataTable(
                name=key,
                data=data,
            )
            data_tables.append(data_table)

        return data_tables
    
    def _load_anndata_layer(
        self,
        adata: AnnData,
        chr: List[str],
        pos: List[int],
        data_key: Optional[str] = "X",
    ) -> Optional[Dataset]:
        """
        Load anndata layer as a :class:Dataset object.

        Args:
            adata (:class:`AnnData`): Annotated data object to load.
            chr: list of chromosome names for each bin.
            pos: list of chromosome positions for each bin.
            data_key (:obj:`str`, optional): Data key to load, default to "X". The data
                key must be in :attr:`adata.X` or :attr:`adata.layers``.
        Returns:
            :class:`Dataset`: Dataset object loaded.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("adata must be an AnnData object.")

        if data_key == "X":
            data = adata.X
        elif data_key in adata.layers:
            data = adata.layers[data_key]
        elif data_key in adata.obsm:
            data = adata.obsm[data_key]
        else:
            logger.warning(f"Data key {data_key} not found, skip loading.")
            return None

        tokenized_data = self._tokenize(data, chr, pos)

        return Dataset.from_dict(tokenized_data)

    def _tokenize(
        self,
        data: Union[np.ndarray, csr_matrix],
        chr: List[str],
        pos: List[int],
    ) -> Dict[str, List]:
        """
        Tokenize the data with the given vocabulary.
        
        Args:
            data (np.ndarray or spmatrix): Data to be tokenized.
            chr: list of chromosome names for each bin.
            pos: list of chromosome positions for each bin.
        Returns:
            Dict[str, List]: Tokenized data.
        """
        if not isinstance(data, (np.ndarray, csr_matrix)):
            raise ValueError("data must be a numpy array or sparse matrix.")

        if isinstance(data, np.ndarray):
            zero_ratio = np.sum(data == 0) / data.size
            if zero_ratio > 0.85:
                logger.debug(
                    f"{zero_ratio*100:.0f}% of the data is zero, "
                    "auto convert to sparse matrix before tokenizing."
                )
                data = csr_matrix(data)  # process sparse matrix actually faster

        # remove zero rows
        if self.settings.remove_zero_rows:
            if isinstance(data, np.ndarray) and data.size > 1e9:
                logger.warning(
                    "Going to remove zero rows from a large ndarray data. This "
                    "may take long time. If you want to disable this, set "
                    f"`remove_zero_rows` to False in {self.__name__}.settings."
                )
            if isinstance(data, csr_matrix):
                data = data[data.getnnz(axis=1) > 0]
            else:
                data = data[~np.all(data == 0, axis=1)]

        n_rows = data.shape[0]
        chr_array = np.array(chr)
        pos_array = np.array(pos) 

        if isinstance(data, csr_matrix):
            indptr = data.indptr
            indices = data.indices

            tokenized_data = {"id": [], "chr": [], "pos": []}
            tokenized_data["id"] = list(range(n_rows))
            for i in range(n_rows):  # ~2s/100k cells
                row_indices = indices[indptr[i] : indptr[i + 1]]
                row_new_chr = chr_array[row_indices]
                row_new_pos = pos_array[row_indices]

                tokenized_data["chr"].append(row_new_chr)
                tokenized_data["pos"].append(row_new_pos)
        else:
            tokenized_data = _nparray2mapped_values(data, chr, pos, "numba")  

        return tokenized_data

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> Self:
        """
        Create a DataBank from a directory containing scBank data. **NOTE**: this
        method will automatically check whether md5sum record in the :file:`manifest.json`
        matches the md5sum of the loaded gene vocabulary.

        Args:
            path (Path or str): Directory path.

        Returns:
            DataBank: DataBank instance.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory.")

        data_table_files = [f for f in path.glob("*.datatable.*") if f.is_file()]
        if len(data_table_files) == 0:
            logger.warning(f"Loading empty DataBank at {path} without datatables.")

        db = cls(meta_info=MetaInfo.from_path(path))
        data_format = db.meta_info.on_disk_format
        for data_table_file in data_table_files:
            logger.info(f"Loading datatable {data_table_file}.")
            data_table = DataTable(
                name=data_table_file.name.split(".")[0],
                data=load_dataset(
                    data_format,
                    data_files=str(data_table_file),
                    cache_dir=str(path),
                    split="train",
                ),
            )
            db.update_datatables(new_tables=[data_table])
        return db


    def _validate_data(self) -> None:
        """
        Validate the current DataBank, including checking md5sum, table names, etc.
        """
        if len(self.data_tables) == 0 and self.main_table_key is not None:
            raise ValueError("No data tables found, but main table key is set."
                "Please set main table key to None or add data tables.")

        if len(self.data_tables) > 0:
            if self.main_table_key is None:
                raise ValueError("Main table key can not be empty if non-empty data tables.")
            if self.main_table_key not in self.data_tables.keys(): # if self.main_table_key not in self.data_tables
                raise ValueError("Main table key {self.main_table_key} not found in data tables.")

    def update_datatables(
        self,
        new_tables: List[DataTable],
        use_names: List[str] = None,
        overwrite: bool = False,
        immediate_save: Optional[bool] = None,
    ) -> None:
        """
        Update the data tables in the DataBank with new data tables.

        Args:
            new_tables (list of :class:`DataTable`): New data tables to update.
            use_names (list of :obj:`str`): Names of the new data tables to use.
                If not provided, will use the names of the new data tables.
            overwrite (:obj:`bool`): Whether to overwrite the existing data tables.
            immediate_save (:obj:`bool`): Whether to save the data immediately after
                updating. Will save to :attr:`self.meta_info.on_disk_path`. If not
                provided, will follow :attr:`self.settings.immediate_save` instead.
                Default to None.
        """
        if not isinstance(new_tables, list) or not all(
            isinstance(t, DataTable) for t in new_tables
        ):
            raise ValueError("new_tables must be a list of DataTable.")

        if use_names is None:
            use_names = [t.name for t in new_tables]
        else:
            if len(use_names) != len(new_tables):
                raise ValueError("use_names must have the same length as new_tables.")

        if not overwrite:
            overlaps = set(use_names) & set(self.data_tables.keys())
            if len(overlaps) > 0:
                raise ValueError(
                    f"Data table names {overlaps} already exist in the DataBank. "
                    "Please set overwrite=True if replacing the existing data table."
                )

        if immediate_save is None:
            immediate_save = self.settings.immediate_save

        for dt, name in zip(new_tables, use_names):
            self.data_tables[name] = dt

        self._validate_data()
        if self.settings.immediate_save:
            self.sync(["data_tables"])

    def sync(self, 
             attr_keys: Union[List[str], str, None] = None,
             suffix: Optional[str] = None) -> None:
        """
        Sync the current DataBank to a data directory, including, save the updated
        data/vocab to files, update the meta info and save to files.
        **NOTE**: This will overwrite the existing data directory.

        Args:
            attr_keys (list of :obj:`str`): List of attribute keys to sync. If None, will
                sync all the attributes with tracked changes.
        """
        if attr_keys is None:
            attr_keys = ["meta_info", "data_tables"]
        elif isinstance(attr_keys, str):
            attr_keys = [attr_keys]

        # TODO: implement. Remeber particularly to update md5 in metainfo when
        # updating the gene vocabulary.

        on_disk_path = self.meta_info.on_disk_path
        data_format = self.meta_info.on_disk_format
        if "meta_info" in attr_keys:
            self.meta_info.save(on_disk_path, suffix=suffix)
        if "data_tables" in attr_keys:
            for data_table in self.data_tables.values():
                if suffix is not None and isinstance(suffix, str):
                    if data_table.name == self.main_table_key:
                        save_to = on_disk_path / f"datatable.{suffix}.{data_format}"
                    else:
                        save_to = on_disk_path / f"datatable.{data_table.name}.{suffix}.{data_format}"
                else:
                    logger.info("No suffix provided, saving to default.")
                    if data_table.name == self.main_table_key:
                        save_to = on_disk_path / f"datatable.{data_format}"
                    else:
                        save_to = on_disk_path / f"datatable.{data_table.name}.{data_format}"
                logger.info(f"Saving data table {data_table.name} to {save_to}.")
                data_table.save(
                    path=save_to,
                    format=data_format,
                )

def _nparray2mapped_values(
    data: np.ndarray,
    chr: List[str],
    pos: List[int],
    mode: Literal["plain", "numba"] = "plain",
) -> Dict[str, List]:
    """
    Convert a numpy array to mapped values. Only include the non-zero values.

    Args:
        data (np.ndarray): Data matrix.
        chr: list of chromosome names for each bin.
        pos: list of chromosome positions for each bin.
        mode (Literal["plain", "numba"]): Mode to use for conversion.

    Returns:
        Dict[str, List]: Mapping from column name to list of values.
    """
    if mode == "plain":
        convert_func = _nparray2indexed_values
    elif mode == "numba":
        convert_func = _nparray2indexed_values_numba
    else:
        raise ValueError(f"Unknown mode {mode}.")
    tokenized_data = {}
    row_ids, chr_, pos_ = convert_func(data, chr, pos)

    tokenized_data["id"] = row_ids
    tokenized_data["chr"] = chr_
    tokenized_data["pos"] = pos_
    return tokenized_data


def _nparray2indexed_values(
    data: np.ndarray,
    chr: List[str],
    pos: List[int],
) -> Tuple[List, List, List]:
    """
    Convert a numpy array to indexed values. Only include the non-zero values.

    Args:
        data (np.ndarray): Data matrix.
        chr: list of chromosome names for each bin.
        pos: list of chromosome positions for each bin.

    Returns:
        Tuple[List, List, List]: Row IDs, chromosome names, and chromosome positions.
    """
    row_ids, chr_ls, pos_ls = [], [], []
    for i in range(len(data)):  # TODO: accelerate with numba? joblib?
        row = data[i]
        idx = np.nonzero(row)[0]
        chr_ = chr[idx]
        pos_ = pos[idx]

        row_ids.append(i)
        chr_ls.append(chr_)
        pos_ls.append(pos_)

    return row_ids, chr_ls, pos_ls


from numba import jit, njit, prange

@njit(parallel=True)
def _nparray2indexed_values_numba(
    data: np.ndarray,
    chr: List[str],
    pos: List[int],
) -> Tuple[List, List, List]:
    """
    Convert a numpy array to indexed values. Only include the non-zero values.
    Using numba to accelerate.

    Args:
        data (np.ndarray): Data matrix.
        chr: list of chromosome names for each bin.
        pos: list of chromosome positions for each bin.

    Returns:
        Tuple[List, List, List]: Row IDs, column indices, and values.
    """
    row_ids, chr_ls, pos_ls = (
        [1] * len(data),
        [np.empty(0, dtype=np.int64)] * len(data),
        [np.empty(0, dtype=data.dtype)] * len(data),
    )  # placeholders
    for i in prange(len(data)):
        row = data[i]
        idx = np.nonzero(row)[0]
        chr_ = chr[idx]
        pos_ = pos[idx]

        row_ids[i] = i
        chr_ls[i] = chr_
        pos_ls[i] = pos_

    return row_ids, chr_ls, pos_ls


