
import os
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader
import torch
import datasets
from transformers import AutoTokenizer
import nltk
import pandas as pd
from models import HypOnly
from train_biased_models import get_infersent_embedder


# %%
dataset = datasets.load_dataset('snli',split='validation')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

# %%
def generate_data():
    nltk.download('punkt')
    batch_size = 64
    K = 100000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load('base_model_params.pt', map_location=device)['model_state_dict']
    model = HypOnly(4096, 4096, 3).to(device)
    model.load_state_dict(model_state)
    model.eval()
    dev_set = datasets.load_dataset('snli', split='validation').with_format('torch', device=device) \
        .filter(lambda ex: ex['label'] != -1)
    # train_loader = train_set.map(lam  bda x: print(x['hypothesis'][0]), batched=False)
    print(dev_set.info.features)
    sampler = BatchSampler(RandomSampler(dev_set), batch_size=20, drop_last=False)
    dev_loader = DataLoader(dev_set, sampler=sampler)
    infer_embedder = get_infersent_embedder(K)

    folder = 'results'
    os.makedirs(folder,exist_ok=True)
    i=0
    correct=0
    for batch in (dev_loader):
        x = torch.from_numpy(infer_embedder.encode(list(map(lambda x: x[0], batch['hypothesis'])))).to(device)
        y = torch.squeeze(batch['label'])
        model.zero_grad()
        y_hat = model.forward(x)
        y_hat = torch.argmax(y_hat, dim=1)
        correct += (y_hat == y).float().sum()
        with open(f'{folder}/xs_{i}.csv','w') as f:
            f.write('\n'.join(map(lambda x: x[0], batch['hypothesis'])))
        pd.DataFrame(y_hat).to_csv(f'{folder}/ys_{i}.csv',index=False)
        i+=1
        break
    accuracy = 100 * correct / len(dev_set)


    print("Accuracy = {}".format(accuracy))
    with open('results.txt', 'w') as f:
        f.write("Accuracy = {}".format(accuracy))
        # print(loss)

# %%
generate_data()

# %%


# %%


# %%



