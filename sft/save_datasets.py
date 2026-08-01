from datasets import load_dataset, DatasetDict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'all'

# for i in ['point', 'group', 'pair']:
#     dataset = load_dataset('json', data_files=str(DATA_DIR / f'train-small-{i}.jsonl'))

#     data_dict = DatasetDict({
#         'train_sft': dataset['train']
#     })
#     data_dict.save_to_disk(str(DATA_DIR / f'train_data-small-{i}'))

#     print(data_dict['train_sft'])


# from datasets import load_dataset, load_from_disk
# datasets = load_from_disk(str(DATA_DIR / 'train_data-point'))

# print(datasets['train_sft']['messages'][0])


# for i in ["mix"]:
#     dataset = load_dataset('json', data_files=str(DATA_DIR / f'train-{i}.jsonl'))

#     data_dict = DatasetDict({
#         'train_sft': dataset['train']
#     })
#     data_dict.save_to_disk(str(DATA_DIR / f'train_data-{i}'))

#     print(data_dict['train_sft'])




from datasets import load_dataset, load_from_disk

dataset = load_from_disk(str(DATA_DIR / 'train_data-mix'))
print(dataset['train_sft']['messages'][0])

# import json
# with open(DATA_DIR / 'train-mix.jsonl', 'r') as f:
#     for line in f.readlines():
#         i = json.loads(line)
#         print(i)
#         break
