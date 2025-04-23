from datasets import Dataset, load_dataset, concatenate_datasets

cls_prefix_datatable = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cls_prefix_data.parquet"
cache_dir = "/lustre/project/Stat/s1155184322/datasets/atacFormer/HuBMAP/heart/cache"

dataset = load_dataset(
            "parquet",
            data_files=str(cls_prefix_datatable),
            split="train",
            cache_dir=str(cache_dir),
        )


dataset[100]["chr_id"][0]
dataset[100]["pos_id"][0]

dataset[100]["chr_id"][-23:]
dataset[100]["pos_id"][-23:]