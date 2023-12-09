import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoTokenizer
import os

from .bertweet_util import normalizeTweet

# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# os.chdir(dname)


class SemEval2016Dataset(Dataset):
    '''
    Dataset containing semeval data. Currently returns tweet as token ids,
    tokenizer-generated attention mask, target as an id (see __init__), and
    stance as an id (see __init__).

    To access raw data instead of tokenized data, probably easiest to just
    access the Dataframe directly:
        >>> dataset = SemEval2016Dataset(...)
        >>> df = dataset.df
    '''
    def __init__(self, csv_file, tokenizer, max_length):
        super().__init__()
        stance2id = {
            'AGAINST': 0,
            'FAVOR': 1,
            'NONE': 2
        }
        target2id = {
            'Atheism': 0,
            'Climate Change is a Real Concern': 1,
            'Donald Trump': 2,
            'Feminist Movement': 3,
            'Hillary Clinton': 4,
            'Legalization of Abortion': 5
        }
        col_names = ['tweet', 'target', 'stance', 'opinion towards', 'sentiment']

        df = pd.read_csv(csv_file, encoding='unicode_escape', lineterminator='\r',
                         header=0, names = col_names)
        df['stance'] = df['stance'].apply(lambda x: stance2id[x])
        df['target'] = df['target'].apply(lambda x: target2id[x])
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        tweet = self.df['tweet'].iloc[ix]
        tweet = preprocess_tweet(tweet) # preprocess tweet (see below)
        target = self.df['target'].iloc[ix]
        stance = self.df['stance'].iloc[ix]
        encoding = self.tokenizer(tweet,
                                  return_tensors='pt',
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True)
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        return input_ids, attention_mask, target, stance

def preprocess_tweet(tweet):
    '''
    Given tweet as string, perform tweet-specific preprocessing as described
    here: https://github.com/VinAIResearch/BERTweet#-normalize-raw-input-tweets
        * translate emojis into text
        * convert @'s and URLs into special tokens

    Example:
        The feds are still planning to put a woman on the $10 dollar bill 💪 #abouttime https://www.google.com/ @Trump
        => The feds are still planning to put a woman on the $ 10 dollar bill :flexed_biceps: #abouttime HTTPURL @USER
    '''
    return normalizeTweet(tweet)

def get_semeval_dataset(tokenizer='vinai/bertweet-large', max_length=128, train=True):
    '''
    Returns a pandas Dataframe containing semeval data.
    Header: [tweet, target, stance, opinion towards, sentiment]

    Args:
        tokenizer: string for hugginface tokenizer
        max_length: max token length
        train: bool for accessing either train or test data
    '''
    if train:
        csv_file = './semeval-2016/train.csv'
    else:
        csv_file = './semeval-2016/test.csv'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    return SemEval2016Dataset(csv_file, tokenizer, max_length)

def get_semeval_data_loader(batch_size=128, shuffle=True,
                            tokenizer='vinai/bertweet-large', max_length=128,
                            train=True):
    '''
    Returns a PyTorch DataLoader for serving semeval data. See below/jupyter
    notebook for example usage.
    Shapes: (B = batch_size, T = max_length)
    input_ids: BxT
    attention_mask: BxT
    target: B
    stance: B

    Args:
        train: bool for accessing either train or test data
    '''
    data = get_semeval_dataset(tokenizer, max_length, train)
    return DataLoader(data, batch_size, shuffle)



if __name__ == '__main__':
    # dataset = get_semeval_dataset()
    # print(dataset[0])

    # loader = get_semeval_data_loader(batch_size=4)
    # for ix, (input_ids, attention_mask, target, stance) in enumerate(loader):
    #     # print(f'{input_ids=}')
    #     # print(f'{attention_mask=}')
    #     print(f'{target=}')
    #     print(f'{stance=}')
    #     print()
    #     if ix == 2:
    #         break

    tweet = 'The feds are still planning to put a woman on the $10 dollar bill 💪 #abouttime #equalityforall #historychanged https://www.google.com/ @Trump'
    print(tweet)
    print(preprocess_tweet(tweet))