# modified from https://github.com/bowang-lab/scGPT/blob/integrate-huggingface-model/examples/pretrain.py

import os
import sys
import argparse
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from datasets import Dataset, load_dataset, concatenate_datasets

import scgpt as scg
from scgpt.utils import MainProcessOnly
from scgpt import logger

sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from data.databank import DataBank
from model.model import TransformerModel
from model.data_collator import DataCollator
from model.utils import chr_pos_to_idx

# torch.autograd.set_detect_anomaly(True)

sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = "debug"
scg.utils.set_seed(42)

# argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data-source",
    type=str,
    required=True,
    help="The name of the data source, or the path to the data file.",
)
parser.add_argument(
    "-e",
    "--dna-emb-source",
    type=str,
    required=True,
    help="The directory containing the DNA embedding table.",
)
parser.add_argument(
    "-a",
    "--atac-bin-source",
    type=str,
    required=True,
    help="The directory containing the ATAC bin list and the total counts.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="The directory to save the trained model and the results.",
)
parser.add_argument(
    "--load-model",
    type=str,
    default=None,
    help="The directory containing the model and configs to load and continue training.",
)
parser.add_argument(
    "--use-memmap",
    type=bool,
    default=False,
    help="Whether to use numpy memmap to load the DNA embedding table. Default is False.",
)

# settings for data
parser.add_argument(
    "--valid-size-or-ratio",
    type=float,
    default=0.1,
    help="The ratio or size of the validation set size if split the dataset. "
    "If value is between 0 and 1, will be parsed as the ratio. If value is "
    "greater than 1 and be an integer, will be parsed as the size. If value "
    "is 0, will not split the dataset.",
)
parser.add_argument(
    "--grad-accu-steps",
    type=int,
    default=1,
    help="The number of gradient accumulation steps. Default is 1.",
)
parser.add_argument(
    "--max-seq-len",
    type=int,
    default=5000,
    help="The maximum length of the sequence. Default is 5000. The actual used "
    "max length would be the minimum of this value and the length of the longest "
    "sequence in the data.",
)
parser.add_argument(
    "--cls-value",
    type=int,
    default=0,
    help="The value used for <cls>. Default is 0.",
)
parser.add_argument(
    "--mask-value",
    type=int,
    default=-1,
    help="The value used for masking. Default is -1.",
)
parser.add_argument(
    "--eos-value",
    type=int,
    default=-2,
    help="The value used for <eos> (end of sentence). Default is -2.",
)
parser.add_argument(
    "--pad-value",
    type=int,
    default=-3,
    help="The value used for padding. Default is -3.",
)
parser.add_argument(
    "--mask-ratio",
    type=float,
    default=0.15,
    help="The ratio of masked values in the training data. Default is 0.15.",
)
parser.add_argument(
    "--trunc-by-sample",
    action="store_true",
    help="Whether to truncate the input by sampling rather than cutting off if "
    "sequence length > max_seq_length. Default is False.",
)

# settings for training
parser.add_argument(
    "--local-rank",
    type=int,
    default=-1,
    help="The local rank of the process for using the torch.distributed.launch "
    "utility. Will be -1 if not running in distributed model.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="The batch size for training. Default is 16.",
)
parser.add_argument(
    "--eval-batch-size",
    type=int,
    default=32,
    help="The batch size for evaluation. Default is 32.",
)

# settings for training
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="The number of epochs for training.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="The learning rate for training. Default is 1e-3.",
)
parser.add_argument(
    "--scheduler-interval",
    type=int,
    default=100,
    help="The interval iterations for updating the learning rate. Default is 100. "
    "This will only be used when warmup-ratio is 0.",
)
parser.add_argument(
    "--scheduler-factor",
    type=float,
    default=0.99,
    help="The factor for updating the learning rate. Default is 0.99. "
    "This will only be used when warmup-ratio is 0.",
)
parser.add_argument(
    "--warmup-ratio-or-step",
    type=float,
    default=0.1,
    help="The ratio of warmup steps out of the total training steps. Default is 0.1. "
    "If warmup-ratio is above 0, will use a cosine scheduler with warmup. If "
    "the value is above 1, will use it as the number of warmup steps.",
)
parser.add_argument(
    "--no-cls",
    action="store_true",
    help="Whether to deactivate the classification loss. Default is False.",
)
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to train in automatic mixed precision. Default is False.",
)
parser.add_argument(
    "--fast-transformer",
    type=bool,
    default=True,
    help="Whether to use the fast transformer. Default is True.",
)
parser.add_argument(
    "--weight-cls",
    type=float,
    default=0.5,
    help="The weight for cls prediction. Default is 0.5.",
)
parser.add_argument(
    "--weight-token",
    type=float,
    default=0.1,
    help="The weight for token prediction in accuracy calculation. Default is 0.1.",
)
parser.add_argument(
    "--weight-masked",
    type=float,
    default=0.1,
    help="The weight for masked position prediction in accuracy calculation. Default is 0.1.",
)

# settings for model
parser.add_argument(
    "--nlayers",
    type=int,
    default=12,
    help="The number of layers for the transformer. Default is 12."
)
parser.add_argument(
    "--nheads",
    type=int,
    default=8,
    help="The number of heads for the transformer. Default is 8."
)
parser.add_argument(
    "--embsize",
    type=int,
    default=512,
    help="The embedding size for the transformer. Default is 512."
)
parser.add_argument(
    "--d-hid",
    type=int,
    default=512,
    help="Dimension of the feedforward network model in the transformer."
    "Default is 512.",
)
parser.add_argument(
    "--bottoleneck-dim",
    type=int,
    default=64,
    help="The bottleneck dim for bin embedding. Default is 64.",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0,
    help="The dropout rate. Default is 0.",
)
parser.add_argument(
    "--n-layers-cls",
    type=int,
    default=3,
    help="The number of layers for the classification network, including the "
    "output layer. Default is 3.",
)
parser.add_argument(
    "--dna-emb-dim",
    type=int,
    default=512,
    help="Dimension of the DNA embedding vector. Default is 512.",
)
parser.add_argument(
    "--use-dna-encoder",
    type=bool,
    default=False,
    help="Whether to use DNA encoder. Default is False.",
)
parser.add_argument(
    "--nlayers-dna-enc",
    type=int,
    default=2,
    help="The number of layers for the DNA embedding vector encoder. Default is 2.",
)

# settings for logging
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="The interval for logging. Default is 100.",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=1000,
    help="The interval for saving the model. Default is 1000.",
)

if scg.utils.isnotebook():
    args = parser.parse_args(
        args=[
            "-d",
            "/lustre/project/Stat/s1155184322/datasets/atacGPT/HuBMAP/heart/cls_prefix_data.parquet",
            "-e"
            "/lustre/project/Stat/s1155184322/datasets/atacGPT/dna_emb_table.npy",
            "-a",
            "/lustre/project/Stat/s1155184322/datasets/atacGPT/var_open_cells_23chr.txt",
            "-s",
            "./save/tmp",
            "--batch-size",
            "16",
            "--max-seq-len",
            "5000",
            "--mask-ratio",
            "0",
            "--dna-emb-dim",
            "1280",
            "--use-dna-encoder",
            "True"
            "--trunc-by-sample",
            "--no-cls",
            "--fp16",
        ]
    )
else:
    args = parser.parse_args()


# args.local_rank = os.environ['LOCAL_RANK']

# show the arguments
print(args)

USE_CLS = not args.no_cls

IS_DATA_PARALLEL = args.local_rank != -1
if IS_DATA_PARALLEL:
    # These two lines is to solve issue #1 based on the suggestion from
    # https://discuss.pytorch.org/t/94382
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        rank=args.local_rank,
        timeout=timedelta(hours=10),
    )
    # specify device 0 since the CUDA_VISIBLE_DEVICES is set to one GPU
    # https://discuss.pytorch.org/t/67488/4
    device = torch.device("cuda:0")
    n_gpu = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    logger.info(
        f"device: {device} in world size {world_size}, "
        f"visible gpu(s): {os.environ['CUDA_VISIBLE_DEVICES']}/{n_gpu}"
    )
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(args.save_dir)
if args.local_rank in [0, -1]:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    # copy all uncommitted changes to the save dir
    os.system(
        f"git diff > {str(save_dir / 'git_diff_')}{scg.utils.get_git_commit()}.diff"
    )
if IS_DATA_PARALLEL:
    torch.distributed.barrier()

scg.utils.add_file_handler(logger, save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Current git commit: {scg.utils.get_git_commit()}")

writer = SummaryWriter(log_dir=save_dir / "tensorboard")
if IS_DATA_PARALLEL:
    writer = MainProcessOnly(writer)


# append <cls> at the beginning of the sequence
def _map_append_cls(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "chr": [args.cls_value] + example["chr"],
            "pos": [args.cls_value] + example["pos"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )

    return dataset


# Load data
# load everything from the data source
if args.data_source.endswith("atacGPT"): # TODO: atacGPT should be changed accordingly
    raw_dataset_list = []
    DATA_LIST = [f for f in os.listdir(args.data_source) 
                     if os.path.isdir(os.path.join(args.data_source, f))]
    
    for database in DATA_LIST:
        TISSUE_LIST = [f for f in os.listdir(os.path.join(args.data_source, database)) 
                       if os.path.isdir(os.path.join(args.data_source, database, f))]
    
        root_data_source = Path(args.data_source) / database
        for tissue in TISSUE_LIST:
            tissue_data_path = root_data_source / tissue
            cls_prefix_datatable = tissue_data_path / "cls_prefix_data.parquet"
            cache_dir = tissue_data_path / "cache"
            
            tissue_dataset = load_dataset(
                "parquet",
                data_files=str(cls_prefix_datatable),
                split="train",
                cache_dir=str(cache_dir),
            )
            logger.info(f"Loaded {tissue} examples from {cls_prefix_datatable}")
            raw_dataset_list.append(tissue_dataset)
        print("merging dataset...")
        raw_dataset = concatenate_datasets(raw_dataset_list)
        print("done merging dataset")

# load from a single .scb file
elif Path(args.data_source).is_dir() and args.data_source.endswith(".scb"):
    # the large-scale data structure
    db = DataBank.from_path(args.data_source)
    raw_dataset = db.main_data.data

    if USE_CLS:
        # load or make the dataset w/ <cls> appended at the beginning
        cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
        if not cls_prefix_datatable.exists():
            if args.local_rank in [0, -1]:
                raw_dataset = _map_append_cls(raw_dataset)
                raw_dataset.to_parquet(cls_prefix_datatable)
            if IS_DATA_PARALLEL:
                torch.distributed.barrier()  # wait for the mapping to finish
        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=args.data_source,
        )
        logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
      
# collection of parquet files
elif Path(args.data_source).is_dir():
    parquet_files = [str(f) for f in Path(args.data_source).glob("*.parquet")]
    cache_dir = Path(args.data_source).parent / "cache"
    
    if USE_CLS:
        # load or make the dataset w/ <cls> appended at the beginning
        cls_prefix_datatable = Path(args.data_source) / "cls_prefix_data.parquet"
        if not cls_prefix_datatable.exists():
            if args.local_rank in [0, -1]:
                logger.info(f"Rank {args.local_rank}: Preparing dataset")
                raw_dataset = load_dataset(
                    "parquet",
                    data_files=parquet_files,
                    split="train",
                    cache_dir=str(cache_dir),
                )
                raw_dataset = _map_append_cls(raw_dataset)
                raw_dataset.to_parquet(str(cls_prefix_datatable))
            if IS_DATA_PARALLEL:
                torch.distributed.barrier()  # wait for the mapping to finish
        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
        )
        logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
        
# load from an adata file
elif Path(args.data_source).is_file() and args.data_source.endswith(".h5ad"):
    adata = sc.read(args.data_source, cache=True)
    # Specific the required column names, when loading the data the first time.
    # Store the column names for later use.
    (
        celltype_col,
        str_celltype_col,
        gene_col,
        batch_key,
    ) = scg.utils.find_required_colums(
        adata,
        id=args.data_source,
        configs_dir=Path(args.data_source),
    )
    if celltype_col is None:
        celltype_col = "int" + str_celltype_col
        adata.obs[celltype_col] = scg.utils.category_str2int(adata.obs[str_celltype_col])
        
# load data directly from the cls_prefix_data.parquet
elif args.data_source.endswith("cls_prefix_data.parquet"):
    cache_dir = Path(args.data_source).parent / "cache"
    cls_prefix_datatable = Path(args.data_source)
    raw_dataset = load_dataset(
                "parquet",
                data_files=str(cls_prefix_datatable),
                split="train",
                cache_dir=str(cache_dir),
            )
    logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
  
# Using test data      
elif args.data_source == "test":  
    raw_dataset = Dataset.from_dict(
        {
            "id": [1] * 300,
            "chr": [[1, 2, 3]] * 300,
            "pos": [[1, 2, 3]] * 300,
        }
    )


# load model from the model directory
if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    if args.cls_value != model_configs["cls_value"]:
        logger.warning(
            f"The cls value in the model directory to load ({model_dir}) "
            "does not match the current pad token. Be careful if this is not expected."
        )
    if args.mask_value != model_configs["mask_value"]:
        logger.warning(
            f"The mask value in the model directory to load ({model_dir}) "
            "does not match the current pad value. Be careful if this is not expected."
        )
    if args.eos_value != model_configs["eos_value"]:
        logger.warning(
            f"The eos value in the model directory to load ({model_dir}) "
            "does not match the current pad value. Be careful if this is not expected."
        )
    if args.pad_value != model_configs["pad_value"]:
        logger.warning(
            f"The pad value in the model directory to load ({model_dir}) "
            "does not match the current pad token. Be careful if this is not expected."
        )
    logger.info(
        f"Resume model from {model_file}, the model args will be overridden the "
        f"config {model_config_file}."
    )
    args.embsize = model_configs["embsize"]
    args.nheads = model_configs["nheads"]
    args.d_hid = model_configs["d_hid"]
    args.nlayers = model_configs["nlayers"]
    args.n_layers_cls = model_configs["n_layers_cls"]

    # resave the args with the new values
    if args.local_rank in [0, -1]:
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

if IS_DATA_PARALLEL:
    torch.distributed.barrier()  # wait for saving all the files

# data processing
# convert format to return torch.tensor
raw_dataset = raw_dataset.with_format("torch")

# split train and validation set
raw_dataset = raw_dataset.train_test_split(test_size=args.valid_size_or_ratio, shuffle=True)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
logger.info(f"train set number of samples: {len(train_dataset)}, ")
logger.info(f"valid set number of samples: {len(valid_dataset)}, ")

# data collator for online padding and sampling
# make separate two types of input and output
collator = DataCollator(
    do_padding=True if args.max_seq_len is not None else False,
    pad_value=args.pad_value,
    eos_value=args.eos_value,
    mask_value=args.mask_value,
    do_mlm=True,
    mlm_probability=args.mask_ratio,
    max_length=args.max_seq_len,
    sampling=args.trunc_by_sample,
)

# TODO: try batch sampler, train_sampler = BatchSampler()
train_sampler = (
    DistributedSampler(train_dataset)
    if IS_DATA_PARALLEL
    else RandomSampler(train_dataset)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), args.batch_size),
    pin_memory=True,
    prefetch_factor=4,
)
valid_sampler = (
    DistributedSampler(valid_dataset, shuffle=False)
    if IS_DATA_PARALLEL
    else SequentialSampler(valid_dataset)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.eval_batch_size,
    sampler=valid_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), args.eval_batch_size),
    pin_memory=True,
)

if USE_CLS:
    # TODO: write the og_labels and ct_labels
    og_labels = raw_dataset["organ"]
    ct_labels = raw_dataset["celltypes"]
    num_types_organs = len(set(og_labels))
    num_types_celltypes = len(set(ct_labels))
    og_labels = np.array(og_labels)
    ct_labels = np.array(ct_labels)
    
# Prepare the DNA embedding table and ATAC bin list
bin_table = pd.read_table(args.atac_bin_source, header=None)
bin_ls = bin_table.iloc[:, 0].tolist()
bin_total_counts = bin_table.iloc[:, 1].tolist()

# get the number of bins for each chromosome
num_bins_list = [] 
for chr in [str(i) for i in range(1, 23)] + ["X"]:
    num_bins_list.append(len([bin_name for bin_name in bin_ls if bin_name.split(":")[0]==chr]))

# the first row of the DNA embedding table should be zero vector
if args.use_memmap:
    dna_emb_table = np.memmap(args.dna_emb_source, dtype='float16', mode='r', shape=(len(bin_ls) + 1, args.dna_emb_dim))
else:
    dna_emb_table = np.load(args.dna_emb_source, allow_pickle=True)
    dna_emb_table = torch.from_numpy(dna_emb_table).to(torch.float16)

# Create and train model
model = TransformerModel(
    d_model=args.embsize,
    nhead=args.nheads,
    d_hid=args.d_hid,
    nlayers=args.nlayers,
    dropout=args.dropout,
    use_fast_transformer=args.fast_transformer,
    fast_transformer_backend="flash",
    dna_emb_dim=args.dna_emb_dim,
    use_dna_encoder=args.use_dna_encoder,
    nlayers_dna_enc=args.nlayers_dna_enc,
    bottoleneck_dim=args.bottoleneck_dim,
    nlayers_cls=args.n_layers_cls,
    n_cls=(num_types_organs, num_types_celltypes) if USE_CLS else (1, 1),
    num_bins_list=num_bins_list,
)

if args.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
    except:
        from collections import OrderedDict

        params = OrderedDict()
        for key, value in torch.load(model_file).items():
            params[key.replace("module.", "")] = value
        model.load_state_dict(params)
        
model.to(device)
logger.info(model)
if IS_DATA_PARALLEL:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=True,
    )

criterion = nn.CrossEntropyLoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# setup scheduler
if args.warmup_ratio_or_step > 0:
    total_num_batches = len(train_loader) * args.epochs
    warmup_steps = (
        int(total_num_batches * args.warmup_ratio_or_step)
        if args.warmup_ratio_or_step < 1
        else int(args.warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_batches,
        last_epoch=-1,
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.scheduler_interval, gamma=args.scheduler_factor
    )

# amp fp16 training
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)


def train(model: nn.Module, 
          train_loader: DataLoader,
          epoch: int) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    # total loss: mlm loss + cls loss
    # total acc: 0, 1, 2 accuracy
    # total token acc: accuracy for 1
    # mlm acc: accuracy for 2
    # or: organ; ct: celltype
    total_loss, total_mlm_loss, total_cls_loss, total_acc, total_token_acc, total_masked_acc, \
        total_og_acc, total_ct_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    log_interval = args.log_interval
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, data_dict in enumerate(train_loader):
        global_iter = epoch * num_batches + batch

        # load the data on CPU
        input_chr = data_dict["masked_chr"] # (batch_size, seq_len)
        input_pos = data_dict["masked_pos"]
        
        # load DNA embedding on CPU
        if args.use_memmap:
            # the index of the bin for extracting the DNA-seq (seq_len_ < seq_len)
            bin_ids_seq = chr_pos_to_idx(input_chr, input_pos, num_bins_list, special_to_zero=True) # (batch_size, seq_len_)
            bin_ids_seq = bin_ids_seq.view(-1).numpy() # (batch_size * seq_len_)
            dna_emb = torch.tensor(dna_emb_table[bin_ids_seq]) # (batch_size * seq_len_, dna_emb_dim)
        else:
            bin_ids_seq = chr_pos_to_idx(input_chr, input_pos, num_bins_list, special_to_zero=True) # (batch_size, seq_len_)
            bin_ids_seq = bin_ids_seq.view(-1) # (batch_size * seq_len_)
            dna_emb = dna_emb_table[bin_ids_seq]
        
        bin_ids_seq = torch.as_tensor(bin_ids_seq).view(input_chr.size(0), -1) # (batch_size, seq_len_)
        dna_emb = dna_emb.view(bin_ids_seq.size(0), bin_ids_seq.size(1), -1) # (batch_size, seq_len_, dna_emb_dim)
        dna_emb = dna_emb.to(device)
        
        # to device
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        dna_emb = dna_emb.to(device)
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            output_dict = model(
                    seq=dna_emb,
                    data_dict=data_dict,
                    src_key_padding_mask=None,
                    use_cls=USE_CLS,
                    )
            
            predictions = output_dict["predictions"] # list[(batch, num_bins[i], 3) for i in range(23)]
            formulated_targets = output_dict["formulated_targets"] # list[(batch, num_bins[i]) for i in range(23)]
            
            mlm_loss = []
            for c in range(23):
                 mlm_loss_c = criterion(predictions[c].view(-1, 3), formulated_targets[c].view(-1).long())
                 mlm_loss.append(mlm_loss_c)
            
            mlm_loss = torch.stack(mlm_loss).mean()
            writer.add_scalar("train/mlm", mlm_loss, global_iter)
            loss = mlm_loss.clone()
            
            if USE_CLS:
                og_labels = data_dict["organ"]
                ct_labels = data_dict["celltypes"]
                og_loss = criterion_cls(output_dict["og_logits"], og_labels)
                ct_loss = criterion_cls(output_dict["ct_logits"], ct_labels)
                cls_loss = og_loss + ct_loss
                loss = loss + args.weight_cls * cls_loss
                writer.add_scalar("train/cls", cls_loss, global_iter)

            writer.add_scalar("train/loss", loss, global_iter)

        if args.grad_accu_steps > 1:
            loss = loss / args.grad_accu_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if args.grad_accu_steps > 1:
            if batch % args.grad_accu_steps == 0 or batch == num_batches - 1:
                scheduler.step()
                optimizer.zero_grad()
        else:
            scheduler.step()
            optimizer.zero_grad()
            
        # calculate the accuracy
        with torch.no_grad():
            acc, token_acc, token_num, masked_acc, masked_num = 0.0, 0.0, 0.0, 0.0, 0.0
            for c in range(23):
                acc += (predictions[c].argmax(dim=-1) == formulated_targets[c]).float().sum()
                token_c = formulated_targets[c] == 1
                mask_c = formulated_targets[c] == 2
                if token_c.sum() > 0:
                    token_num += token_c.sum()
                    token_acc += ((predictions[c].argmax(dim=-1) == formulated_targets[c]) * token_c).float().sum()
                if mask_c.sum() > 0:
                    masked_num += mask_c.sum()
                    masked_acc += ((predictions[c].argmax(dim=-1) == formulated_targets[c]) * mask_c).float().sum()
            
            acc = acc / sum(num_bins_list) / args.batch_size
            token_acc = token_acc / token_num if token_num > 0 else 0.0
            masked_acc = masked_acc / masked_num if masked_num > 0 else 0.0

            writer.add_scalar("train/acc", acc, global_iter)
            writer.add_scalar("train/token_acc", token_acc, global_iter)
            writer.add_scalar("train/masked_acc", masked_acc, global_iter)
            
            if USE_CLS:
                og_acc = (output_dict["og_logits"].argmax(dim=-1) == og_labels).float().mean()
                ct_acc = (output_dict["ct_logits"].argmax(dim=-1) == ct_labels).float().mean()
                writer.add_scalar("train/og_acc", og_acc, global_iter)
                writer.add_scalar("train/ct_acc", ct_acc, global_iter)

        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_acc += acc
        total_token_acc += token_acc
        total_masked_acc += masked_acc
        total_cls_loss += cls_loss if USE_CLS else 0.0
        total_og_acc += og_acc if USE_CLS else 0.0
        total_ct_acc += ct_acc if USE_CLS else 0.0
        
        if args.local_rank in [0, -1] and batch % log_interval == 0 and batch > 0:
            # Writer logs gradients distribution
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    writer.add_histogram(name + "_grad", param.grad, global_iter)
                    writer.add_histogram(name + "_param", param, global_iter)

            # Log scalar values
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mlm_loss = total_mlm_loss / log_interval
            cur_acc = total_acc / log_interval
            cur_token_acc = total_token_acc / log_interval
            cur_masked_acc = total_masked_acc / log_interval
            cur_cls_loss = total_cls_loss / log_interval if USE_CLS else 0.0
            cur_og_acc = total_og_acc / log_interval if USE_CLS else 0.0
            cur_ct_acc = total_ct_acc / log_interval if USE_CLS else 0.0
            
            # ppl = math.exp(cur_loss)
            logger.info(
            f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
            f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
            f"loss {cur_loss:5.4f} | mlm_loss {cur_mlm_loss:5.4f} |"
            + (f"acc {100 * cur_acc:5.2f}% | ") 
            + (f"token_acc {100 * cur_token_acc:5.2f}% | ")
            + (f"masked_acc {100 * cur_masked_acc:5.2f}% | ")
            + (f"cls {cur_cls_loss:5.4f} | " if USE_CLS else "")
            + (f"og_acc {100 * cur_og_acc:5.2f}% | " if USE_CLS else "")
            + (f"ct_acc {100 * cur_ct_acc:5.2f}% | " if USE_CLS else "")
            )
            writer.add_scalar("lr", lr, global_iter)

            total_loss, total_mlm_loss, total_cls_loss, total_acc, total_token_acc, \
                total_masked_acc, total_og_acc, total_ct_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            start_time = time.time()

        # immediately eval and save
        if batch % args.save_interval == 0 and batch > 0:
            eval_and_save(model, valid_loader, global_iter)
            model.train()  # important, reset to train mode


def evaluate(model: nn.Module, valid_loader: DataLoader) -> Dict[str, torch.Tensor]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for data_dict in valid_loader:
            # load the data on CPU
            input_chr = data_dict["masked_chr"] # (batch_size, seq_len)
            input_pos = data_dict["masked_pos"]
            
            # load DNA embedding on CPU
            if args.use_memmap:
                # the index of the bin for extracting the DNA-seq
                bin_ids_seq = chr_pos_to_idx(input_chr, input_pos, num_bins_list, special_to_zero=True) # (batch_size, seq_len)
                bin_ids_seq = bin_ids_seq.view(-1).numpy() # (batch_size * seq_len)
                dna_emb = torch.tensor(dna_emb_table[bin_ids_seq]) # (batch_size * seq_len, dna_emb_dim)
            else:
                bin_ids_seq = chr_pos_to_idx(input_chr, input_pos, num_bins_list, special_to_zero=True) # (batch_size, seq_len)
                bin_ids_seq = bin_ids_seq.view(-1)
                dna_emb = dna_emb_table[bin_ids_seq]
                
            bin_ids_seq = torch.as_tensor(bin_ids_seq).view(input_chr.size(0), -1) # (batch_size, seq_len_)
            dna_emb = dna_emb.view(bin_ids_seq.size(0), bin_ids_seq.size(1), -1) # (batch_size, seq_len_, dna_emb_dim)
            dna_emb = dna_emb.to(device)
            
            # to device
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            dna_emb = dna_emb.to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                output_dict = model(
                    seq=dna_emb,
                    data_dict=data_dict,
                    src_key_padding_mask=None,
                    use_cls=USE_CLS,
                    )
            
                predictions = output_dict["predictions"] # list[(batch, num_bins[i], 3) for i in range(23)]
                formulated_targets = output_dict["formulated_targets"] # list[(batch, num_bins[i]) for i in range(23)]

                mlm_loss = []
                for c in range(23):
                    mlm_loss_c = criterion(predictions[c].view(-1, 3), formulated_targets[c].view(-1).long())
                    mlm_loss.append(mlm_loss_c)
                
                mlm_loss = torch.stack(mlm_loss).mean()
                loss = mlm_loss.clone()
                
                if USE_CLS:
                    og_labels = data_dict["organ"]
                    ct_labels = data_dict["celltypes"]
                    og_loss = criterion_cls(output_dict["og_logits"], og_labels)
                    ct_loss = criterion_cls(output_dict["ct_logits"], ct_labels)
                    cls_loss = og_loss + ct_loss
                    loss = loss + args.weight_cls * cls_loss
                
            total_loss += loss.item()
            
            # accuracy
            acc, token_acc, token_num, masked_acc, masked_num = 0.0, 0.0, 0.0, 0.0, 0.0
            for c in range(23):
                acc += (predictions[c].argmax(dim=-1) == formulated_targets[c]).float().sum()
                token_c = formulated_targets[c] == 1
                mask_c = formulated_targets[c] == 2
                if token_c.sum() > 0:
                    token_num += token_c.sum()
                    token_acc += ((predictions[c].argmax(dim=-1) == formulated_targets[c]) * token_c).float().sum()
                if mask_c.sum() > 0:
                    masked_num += mask_c.sum()
                    masked_acc += ((predictions[c].argmax(dim=-1) == formulated_targets[c]) * mask_c).float().sum()
            
            acc = acc / sum(num_bins_list) / args.batch_size
            token_acc = token_acc / token_num if token_num > 0 else 0.0
            masked_acc = masked_acc / masked_num if masked_num > 0 else 0.0
            
            if USE_CLS:
                og_acc = (output_dict["og_logits"].argmax(dim=-1) == og_labels).float().mean()
                ct_acc = (output_dict["ct_logits"].argmax(dim=-1) == ct_labels).float().mean()
                
            total_acc += acc + args.weight_masked * masked_acc \
                + args.weight_token * token_acc \
                + args.weight_cls * (og_acc + ct_acc)
            
    total_loss = total_loss / len(valid_loader)
    total_acc = total_acc / len(valid_loader)
    
    return {
        "total_loss": torch.tensor(total_loss, device=device, dtype=torch.float),
        "total_acc": torch.tensor(total_acc, device=device, dtype=torch.float),
    }


def eval_and_save(
    model: nn.Module,
    valid_loader: DataLoader,
    iter_or_epoch: int,
    is_epoch: bool = False,
    save: bool = True,
    epoch_start_time: float = 0.0,
) -> None:
    # perform evaluation in distributed data parallel
    val_loss, val_acc = evaluate(model, valid_loader).values()
    
    if IS_DATA_PARALLEL:
        # gather the results from all the processes
        val_loss_list = [torch.zeros_like(val_loss) for _ in range(world_size)]
        val_acc_list = [torch.zeros_like(val_acc) for _ in range(world_size)]

        torch.distributed.all_gather(val_loss_list, val_loss)
        torch.distributed.all_gather(val_acc_list, val_acc)
        
        val_loss = torch.mean(torch.stack(val_loss_list))
        val_acc = torch.mean(torch.stack(val_acc_list))
        
    val_loss, val_acc = (val_loss.item(), val_acc.item())
    
    if args.local_rank in [0, -1]:
        if is_epoch:
            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {iter_or_epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss {val_loss:5.4f} | acc {val_acc:5.4f} | "
            )
            logger.info(f"{'-' * 89}\n")
            writer.add_scalar("valid/loss", val_loss, iter_or_epoch * len(valid_loader))
            writer.add_scalar("valid/acc", val_acc, iter_or_epoch * len(valid_loader))
        else:
            logger.info(
                f"valid loss {val_loss:5.4f} | acc {val_acc:5.4f} | "
            )
            writer.add_scalar("valid/loss", val_loss, iter_or_epoch)
            writer.add_scalar("valid/acc", val_acc, iter_or_epoch)

        global best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the best model
            logger.info(f"Saving the best model to {args.save_dir}")
            torch.save(
                model.module.state_dict()
                if isinstance(
                    model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else model.state_dict(),
                args.save_dir + "/best_model.pt",
            )

        if save:
            torch.save(
                model.module.state_dict()
                if isinstance(
                    model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
                )
                else model.state_dict(),
                args.save_dir + f"/model-{'ep' if is_epoch else ''}{iter_or_epoch}.pt",
            )
    if IS_DATA_PARALLEL:
        torch.distributed.barrier()


best_val_loss = float("inf")
logger.info("Start training")
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, train_loader, epoch=epoch)
    eval_and_save(model, valid_loader, iter_or_epoch=epoch, is_epoch=True, epoch_start_time=epoch_start_time)

writer.flush()
writer.close()
