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

sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from data.databank import DataBank
from data.vocab import BinVocab
from model.model import TransformerModel
from model.data_collator import DataCollator

sc.set_figure_params(figsize=(4, 4))
sc.settings.verbosity = "debug"
scg.utils.set_seed(42)


# define the logger
import logging
logger = logging.getLogger("atacFormer")
# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
    "-b",
    "--bin-file",
    type=str,
    required=True,
    help="The file to construct bin vocab.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="The directory to save the trained model and the results.",
)
parser.add_argument(
    "--dna-emb-file",
    type=str,
    default=None,
    help="The file containing the DNA embedding table.",
)
parser.add_argument(
    "--load-model",
    type=str,
    default=None,
    help="The directory containing the model and configs to load and continue training.",
)
parser.add_argument(
    "--use-memmap",
    action="store_true",
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
    "--max-input-len",
    type=int,
    default=6800,
    help="The maximum length of the input cell sequence. Default is 6800.",
)
parser.add_argument(
    "--max-masked-len",
    type=int,
    default=3000,
    help="The maximum length of the masked cell sequence. Default is 3000.",
)
parser.add_argument(
    "--masked-ratio",
    type=float,
    default=0.15,
    help="The ratio of masked values in the training data. Default is 0.15.",
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
    "--fp16",
    action="store_true",
    help="Whether to train in automatic mixed precision. Default is False.",
)
parser.add_argument(
    "--no-cls",
    action="store_false",
    dest="use_cls",
    help="Whether to deactivate the classification loss. Default is False.",
)
parser.add_argument(
    "--no-fast-transformer",
    action="store_false",
    dest="fast_transformer",
    help="Whether to disable the fast transformer. Default is True.", # when using --no-fast-transformer, the fast-transformer is False
)
parser.add_argument(
    "--loss-weight",
    type=float,
    default=300,
    help="The weight for class 1 and 2 in cross entropy loss calculation. \
    Default is 300 (6e5 tokens in total / 2e3 tokens each cell).",
)
parser.add_argument(
    "--loss-weight-cls",
    type=float,
    default=0.5,
    help="The weight for cls prediction in loss calculation. Default is 0.5.",
)
parser.add_argument(
    "--acc-weight-cls",
    type=float,
    default=0.5,
    help="The weight for cls prediction in accuracy calculation. Default is 0.5.",
)
parser.add_argument(
    "--acc-weight-1",
    type=float,
    default=0.1,
    help="The weight for token prediction in accuracy calculation. Default is 0.1.",
)
parser.add_argument(
    "--acc-weight-2",
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
    "--use-dna-emb",
    action="store_true",
    help="Whether to use DNA embedding. Default is False.",
)
parser.add_argument(
    "--dna-emb-dim",
    type=int,
    default=512,
    help="Dimension of the DNA embedding vector. Default is 512.",
)
parser.add_argument(
    "--use-dna-encoder",
    action="store_true",
    help="Whether to use DNA encoder. Default is False.",
)
parser.add_argument(
    "--nlayers-dna-enc",
    type=int,
    default=2,
    help="The number of layers for the DNA embedding vector encoder. Default is 2.",
)
parser.add_argument(
    "--decoder-dim",
    type=int,
    default=64,
    help="Dimension of the embedding in the decoder (for both <eos> and bins). Default is 64.",
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
            "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cls_prefix_data.parquet",
            "-b",
            "/lustre/project/Stat/s1155184322/datasets/atacFormer/bins_5k_table_23chr.txt",
            "-s",
            "./save/tmp",
            "--dna-emb-file",
            "/lustre/project/Stat/s1155184322/datasets/atacFormer/dna_emb_table.npy",
            "--batch-size",
            "16",
            "--max-input-len",
            "5000",
            "--max-masked-len",
            "3000",
            "--masked-ratio",
            "0.0",
            "--no-cls",
            "--fp16",
        ]
    )
else:
    args = parser.parse_args()


# args.local_rank = os.environ['LOCAL_RANK']

# show the arguments
print(args)

USE_CLS = args.use_cls

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
if args.local_rank in [0, -1]:
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Current git commit: {scg.utils.get_git_commit()}")

writer = SummaryWriter(log_dir=save_dir / "tensorboard")
if IS_DATA_PARALLEL:
    writer = MainProcessOnly(writer)
    
def _map_append_cls(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "chr_id": [bin_vocab.token_name_to_token("<cls>")[0]] + example["chr_id"],
            "pos_id": [bin_vocab.token_name_to_token("<cls>")[1]] + example["pos_id"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
    )
    return dataset

def _map_append_eos(dataset: Dataset) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "chr_id": example["chr_id"] + [bin_vocab.token_name_to_token(f"<eos_{i}>")[0] for i in range(1, 24)],
            "pos_id": example["pos_id"] + [bin_vocab.token_name_to_token(f"<eos_{i}>")[1] for i in range(1, 24)],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=len(os.sched_getaffinity(0)),
        )
    return dataset

# Load data
# load everything from the data source
if args.data_source.endswith("atacFormer"): # TODO: atacFormer should be changed accordingly
    raw_dataset_list = []
    DATA_BASE = [f for f in os.listdir(args.data_source) 
                     if os.path.isdir(os.path.join(args.data_source, f))]
    
    for database in DATA_BASE:
        DATA_LIST = [f for f in os.listdir(os.path.join(args.data_source, database)) 
                       if os.path.isdir(os.path.join(args.data_source, database, f))]
    
        root_data_source = Path(args.data_source) / database
        for data in DATA_LIST:
            data_path = root_data_source / data
            cls_prefix_datatable = data_path / "cls_prefix_data.parquet"
            cache_dir = data_path / "cache"
            
            dataset = load_dataset(
                "parquet",
                data_files=str(cls_prefix_datatable),
                split="train",
                cache_dir=str(cache_dir),
            )
            if args.local_rank in [0, -1]:
                logger.info(f"Loaded {data} examples from {cls_prefix_datatable}")
            raw_dataset_list.append(dataset)
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
        if args.local_rank in [0, -1]:
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
                raw_dataset = _map_append_eos(raw_dataset)
                raw_dataset.to_parquet(str(cls_prefix_datatable))
            if IS_DATA_PARALLEL:
                torch.distributed.barrier()  # wait for the mapping to finish
        raw_dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
        )
        if args.local_rank in [0, -1]:
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
    if args.local_rank in [0, -1]:
        logger.info(f"Loaded {len(raw_dataset)} examples from {cls_prefix_datatable}")
  
# Using test data      
elif args.data_source == "test":  
    raw_dataset = Dataset.from_dict(
        {
            "id": [1] * 300,
            "chr_id": [[1, 2, 3]] * 300,
            "pos_id": [[1, 2, 3]] * 300,
        }
    )


# load model from the model directory
if args.load_model is not None:
    model_dir = Path(args.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
        
    if args.local_rank in [0, -1]:
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
# load bin vocab
bin_vocab = BinVocab(args.bin_file)
TOTAL_BIN_NUM = sum(bin_vocab.bin_num_dict.values())
    
# convert format to return torch.tensor
raw_dataset = raw_dataset.with_format("torch")

# split train and validation set
raw_dataset = raw_dataset.train_test_split(test_size=args.valid_size_or_ratio, shuffle=True)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
if args.local_rank in [0, -1]:
    logger.info(f"train set number of samples: {len(train_dataset)}, ")
    logger.info(f"valid set number of samples: {len(valid_dataset)}, ")

# data collator for online padding and sampling
# make separate two types of input and output
collator = DataCollator(
    vocab=bin_vocab,
    masked_ratio=args.masked_ratio,
    max_input_len=args.max_input_len,
    max_masked_len=args.max_masked_len,
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
    
# the first 25 rows (<cls>, <pad> and 23 <eos>)) of the DNA embedding table should be zero vector
if args.use_dna_emb:
    if args.use_memmap:
        dna_emb_table = np.memmap(args.dna_emb_file, dtype='float16', mode='r', shape=(len(bin_vocab.vocab), args.dna_emb_dim))
    else:
        dna_emb_table = np.load(args.dna_emb_file, allow_pickle=True)
        dna_emb_table = torch.from_numpy(dna_emb_table).to(torch.float16)

# Create and train model
model = TransformerModel(
    vocab=bin_vocab,
    d_model=args.embsize,
    nhead=args.nheads,
    d_hid=args.d_hid,
    nlayers=args.nlayers,
    dropout=args.dropout,
    use_fast_transformer=args.fast_transformer,
    fast_transformer_backend="flash",
    dna_emb_dim=args.dna_emb_dim,
    use_dna_emb=args.use_dna_emb,
    use_dna_encoder=args.use_dna_encoder,
    nlayers_dna_enc=args.nlayers_dna_enc,
    bottoleneck_dim=args.bottoleneck_dim,
    decoder_dim=args.decoder_dim,
    nlayers_cls=args.n_layers_cls,
    n_cls=(num_types_organs, num_types_celltypes) if USE_CLS else (1, 1),
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
if args.local_rank in [0, -1]:
    logger.info(model)
if IS_DATA_PARALLEL:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=True,
    )

# state weights for the loss function (0, 1, 2)
state_weights = [1, (1 - args.masked_ratio) * args.loss_weight, args.masked_ratio * args.loss_weight]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(state_weights, dtype=torch.float).to(device))
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
        input_ind = data_dict["input_ind"] # (batch_size, seq_len)
        masked_ind = data_dict["masked_ind"]
        
        # load DNA embedding on CPU
        if args.use_dna_emb:
            if args.use_memmap:
                ind_seq = input_ind.clone().view(-1).numpy() # (batch_size * seq_len)
                dna_emb = torch.tensor(dna_emb_table[ind_seq]) # (batch_size * seq_len, dna_emb_dim)
            else:
                ind_seq = input_ind.clone().view(-1) # (batch_size * seq_len)
                dna_emb = dna_emb_table[ind_seq]
            
            dna_emb = dna_emb.view(input_ind.size(0), input_ind.size(1), -1) # (batch_size, seq_len, dna_emb_dim)
        
        # to device
        data_dict = {k: (v.to(device) if v is not None else v) for k, v in data_dict.items()}
        dna_emb = dna_emb.to(device) if args.use_dna_emb else None
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            output_dict = model(
                    data_dict=data_dict,
                    dna_emb=dna_emb,
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
                loss = loss + args.loss_weight_cls * cls_loss
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
            
            acc = acc / TOTAL_BIN_NUM / args.batch_size
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
            input_ind = data_dict["input_ind"] # (batch_size, seq_len)
            masked_ind = data_dict["masked_ind"]
            
            # load DNA embedding on CPU
            if args.use_dna_emb:
                if args.use_memmap:
                    ind_seq = input_ind.clone().view(-1).numpy() # (batch_size * seq_len)
                    dna_emb = torch.tensor(dna_emb_table[ind_seq]) # (batch_size * seq_len, dna_emb_dim)
                else:
                    ind_seq = input_ind.clone().view(-1) # (batch_size * seq_len)
                    dna_emb = dna_emb_table[ind_seq]
                
                dna_emb = dna_emb.view(input_ind.size(0), input_ind.size(1), -1) # (batch_size, seq_len, dna_emb_dim)

            # to device
            data_dict = {k: (v.to(device) if v is not None else v) for k, v in data_dict.items()}
            dna_emb = dna_emb.to(device) if args.use_dna_emb else None

            with torch.cuda.amp.autocast(enabled=args.fp16):
                output_dict = model(
                    data_dict=data_dict,
                    dna_emb=dna_emb,
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
                    loss = loss + args.loss_weight_cls * cls_loss
                
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
            
            acc = acc / TOTAL_BIN_NUM / args.batch_size
            token_acc = token_acc / token_num if token_num > 0 else 0.0
            masked_acc = masked_acc / masked_num if masked_num > 0 else 0.0
            
            if USE_CLS:
                og_acc = (output_dict["og_logits"].argmax(dim=-1) == og_labels).float().mean()
                ct_acc = (output_dict["ct_logits"].argmax(dim=-1) == ct_labels).float().mean()
                
            assert args.acc_weight_1 + args.acc_weight_2 <= 1, "The sum of acc_weight_1 and acc_weight_2 should be less than 1."
            total_acc += (1 - args.acc_weight_1 - args.acc_weight_2) * acc + args.acc_weight_1 * token_acc \
                + args.acc_weight_2 * masked_acc
            if USE_CLS:
                total_acc = (1 - args.acc_weight_cls) * total_acc + args.acc_weight_cls * (og_acc + ct_acc)
            
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
if args.local_rank in [0, -1]:
    logger.info("Start training")
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model, train_loader, epoch=epoch)
    eval_and_save(model, valid_loader, iter_or_epoch=epoch, is_epoch=True, epoch_start_time=epoch_start_time)

writer.flush()
writer.close()
