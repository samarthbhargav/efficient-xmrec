from typing import Dict, List, Tuple

import numpy as np
import torch
import random

from torch.utils.data import DataLoader, Dataset, ConcatDataset

DEBUG = False


class CentralIDBank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """

    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        self.last_user_index = 0
        self.last_item_index = 0
        self.frozen = False

    def freeze(self):
        self.frozen = True

    @property
    def n_items(self):
        return len(self.item_id_index)

    @property
    def n_users(self):
        return len(self.user_id_index)

    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            if self.frozen:
                raise ValueError(f'USER index {user_id} is not valid!')
            self.user_id_index[user_id] = self.last_user_index
            self.last_user_index += 1
        return self.user_id_index[user_id]

    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            if self.frozen:
                raise ValueError(f'Item index {item_id} is not valid!')
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]

    def query_user_id(self, user_index):
        user_index_id = {v: k for k, v in self.user_id_index.items()}
        if user_index in user_index_id:
            return user_index_id[user_index]
        else:
            raise ValueError(f'USER index {user_index} is not valid!')

    def query_item_id(self, item_index):
        item_index_id = {v: k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            raise ValueError(f'ITEM index {item_index} is not valid!')


class MarketTask(Dataset):
    """
    Individual Market data that is going to be wrapped into a metadataset  i.e. MetaMarketDataset

    Wrapper, convert <user, item, rate> Tensor into Pytorch Dataset
    """

    def __init__(self, task_index, user_tensor, item_tensor, target_tensor, market=None):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.task_index = task_index
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.market = market

    def __len__(self):
        return self.user_tensor.size(0)

    def __getitem__(self, index):
        if self.market:
            return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], self.market
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]


class EquallySampledConcatDataset(Dataset):
    """
    Dataset that equally samples from the list of given datasets
    """
    indices: List[Tuple[int, int]]
    datasets: List[MarketTask]

    def __init__(self, datasets: List[MarketTask], n_samples: int, shuffle=True, shuffle_dataset=False):
        super().__init__()
        self.datasets = datasets
        # if True, the dataset idx is also sampled
        self.shuffle_dataset = shuffle_dataset

        # build sampling index
        # this maps from data_idx to [dataset_num, sample_num_in_dataset]
        self.indices = []

        for i, d in enumerate(self.datasets):
            idx = np.arange(len(d))
            if shuffle:
                idx = np.random.permutation(idx)

            assert n_samples <= len(idx)
            idx = idx[:n_samples]

            self.indices.extend([(i, idx[ii]) for ii in range(len(idx))])

        if self.shuffle_dataset:
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)

        # we want the shuffled index
        dataset_idx, sample_idx = self.indices[idx]
        return self.datasets[dataset_idx][sample_idx]


class MAMLTaskGenerator(object):
    """Construct torch dataset"""

    def __init__(self, ratings, market, id_index_bank, item_thr=0, users_allow=None, items_allow=None, sample_df=1):
        """
        args:
            ratings: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'rate']

        """
        self.ratings = ratings
        self.id_index_bank = id_index_bank
        self.market = market

        self.item_thr = item_thr
        self.sample_df = sample_df

        # filter non_allowed users and items
        if users_allow is not None:
            self.ratings = self.ratings[self.ratings['userId'].isin(users_allow)]
        if items_allow is not None:
            self.ratings = self.ratings[self.ratings['itemId'].isin(items_allow)]

        # get item and user pools
        self.user_pool_ids = set(self.ratings['userId'].unique())
        self.item_pool_ids = set(self.ratings['itemId'].unique())

        # replace ids with corrosponding index for both users and items
        self.ratings['userId'] = self.ratings['userId'].apply(lambda x: self.id_index_bank.query_user_index(x))
        self.ratings['itemId'] = self.ratings['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x))

        # get item and user pools (indexed version)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())

        # specify the splits of the data, normalize the vote
        self.user_stats = self._specify_splits()
        self.ratings['rate'] = [self.single_vote_normalize(cvote) for cvote in list(self.ratings.rate)]

        # create negative item samples
        self.negatives_train, self.negatives_valid, self.negatives_test = self._sample_negative(self.ratings)

        # split the data into train, valid, and test
        self.train_ratings, self.valid_ratings, self.test_ratings = self._split_loo(self.ratings)

        if DEBUG:
            print(
                f"{self.market}:: n users: {len(self.user_pool)}, n items:{len(self.item_pool)}, n ratings: {len(self.ratings)}")

    # returns how many training interation for each user has been used
    def get_user_stats(self):
        return self.user_stats

    # adds a new column with each split, and removes the rows below the number of item_thr
    def _specify_splits(self):
        self.ratings = self.ratings.sort_values(['date'], ascending=True)
        self.ratings.reset_index(drop=True, inplace=True)
        by_userid_group = self.ratings.groupby("userId")

        splits = ['remove'] * len(self.ratings)

        user_stats = {}

        for usrid, indice in by_userid_group.groups.items():
            cur_item_list = list(indice)
            if len(cur_item_list) >= self.item_thr:
                train_up_indx = len(cur_item_list) - 2
                valid_up_index = len(cur_item_list) - 1

                sampled_train_up_indx = int(train_up_indx / self.sample_df)

                user_stats[usrid] = len(cur_item_list[:sampled_train_up_indx])

                for iind in cur_item_list[:sampled_train_up_indx]:
                    splits[iind] = 'train'
                for iind in cur_item_list[train_up_indx:valid_up_index]:
                    splits[iind] = 'valid'
                for iind in cur_item_list[valid_up_index:]:
                    splits[iind] = 'test'
        self.ratings['split'] = splits
        self.ratings = self.ratings[self.ratings['split'] != 'remove']
        self.ratings.reset_index(drop=True, inplace=True)

        return user_stats

    # ratings normalization
    def single_vote_normalize(self, cur_vote):
        if cur_vote >= 1:
            return 1.0
        else:
            return 0.0

    def _split_loo(self, ratings):
        train_sp = ratings[ratings['split'] == 'train']
        valid_sp = ratings[ratings['split'] == 'valid']
        test_sp = ratings[ratings['split'] == 'test']
        return train_sp[['userId', 'itemId', 'rate']], valid_sp[['userId', 'itemId', 'rate']], test_sp[
            ['userId', 'itemId', 'rate']]

    def _sample_negative(self, ratings):
        by_userid_group = ratings.groupby("userId")['itemId']
        negatives_train = {}
        negatives_test = {}
        negatives_valid = {}
        for userid, group_frame in by_userid_group:
            pos_itemids = set(group_frame.values.tolist())
            neg_itemids = self.item_pool - pos_itemids

            # neg_itemids_train = random.sample(neg_itemids, min(len(neg_itemids), 1000))
            neg_itemids_train = neg_itemids
            neg_itemids_test = random.sample(neg_itemids, min(len(neg_itemids), 99))
            neg_itemids_valid = random.sample(neg_itemids, min(len(neg_itemids), 99))

            negatives_train[userid] = neg_itemids_train
            negatives_test[userid] = neg_itemids_test
            negatives_valid[userid] = neg_itemids_valid

        return negatives_train, negatives_valid, negatives_test

    def instance_a_market_train_task(self, index, num_negatives, data_frac=1):
        if DEBUG:
            print("train_task", index, self.market)
        """instance train task's torch Dataset"""
        users, items, ratings = [], [], []
        train_ratings = self.train_ratings

        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rate))

            cur_negs = self.negatives_train[int(row.userId)]
            cur_negs = random.sample(cur_negs, min(num_negatives, len(cur_negs)))
            for neg in cur_negs:
                users.append(int(row.userId))
                items.append(int(neg))
                ratings.append(float(0))  # negative samples get 0 rating

        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                             item_tensor=torch.LongTensor(items),
                             target_tensor=torch.FloatTensor(ratings),
                             market=self.market)
        return dataset

    def instance_a_market_train_dataloader(self, index, num_negatives, sample_batch_size, shuffle=True, num_workers=0,
                                           data_frac=1):
        """instance train task's torch Dataloader"""
        dataset = self.instance_a_market_train_task(index, num_negatives, data_frac)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True)

    def instance_a_market_valid_task(self, index, split='valid'):
        if DEBUG:
            print("valid_task", index, self.market, split)
        """instance validation/test task's torch Dataset"""
        cur_ratings = self.valid_ratings
        cur_negs = self.negatives_valid
        if split.startswith('test'):
            cur_ratings = self.test_ratings
            cur_negs = self.negatives_test

        users, items, ratings = [], [], []
        for row in cur_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rate))

            cur_uid_negs = cur_negs[int(row.userId)]
            for neg in cur_uid_negs:
                users.append(int(row.userId))
                items.append(int(neg))
                ratings.append(float(0))  # negative samples get 0 rating

        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                             item_tensor=torch.LongTensor(items),
                             target_tensor=torch.FloatTensor(ratings),
                             market=self.market)
        return dataset

    def instance_a_market_valid_dataloader(self, index, sample_batch_size, shuffle=False, num_workers=0, split='valid'):
        """instance train task's torch Dataloader"""
        dataset = self.instance_a_market_valid_task(index, split=split)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True)

    def get_validation_qrel(self, split='valid'):
        """get pytrec eval version of qrel for evaluation"""
        cur_ratings = self.valid_ratings
        if split.startswith('test'):
            cur_ratings = self.test_ratings
        qrel = {}
        for row in cur_ratings.itertuples():
            cur_user_qrel = qrel.get(str(row.userId), {})
            cur_user_qrel[str(row.itemId)] = int(row.rate)
            qrel[str(row.userId)] = cur_user_qrel
        return qrel


class MetaMarketDataloaders:
    def __init__(self,
                 tasks: Dict[int, MAMLTaskGenerator],
                 sampling_method: str,
                 num_train_negatives: int,
                 batch_size: int,
                 shuffle=True,
                 num_workers=0,
                 pin_memory=False):

        assert sampling_method in {"no_aug", "equal", "concat"}
        if sampling_method == "no_aug":
            assert len(tasks) == 1, "no_aug needs num_tasks=1"

        self.sampling_method = sampling_method
        self.num_tasks = len(tasks)
        self.tasks = tasks
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_train_negatives = num_train_negatives

    def get_split(self, task_idx, split) -> MarketTask:
        if split == "train":
            return self.tasks[task_idx].instance_a_market_train_task(task_idx, num_negatives=self.num_train_negatives)

        else:
            return self.tasks[task_idx].instance_a_market_valid_task(task_idx, split)

    def get_valid_dataloader(self, task_idx, split, batch_size=None, shuffle=False):
        dataset = self.get_split(task_idx, split)
        if DEBUG:
            print(f"task idx: {task_idx} split: {split}:: n-ratings {len(dataset)}")
        return DataLoader(dataset,
                          batch_size=self.batch_size if batch_size is None else batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def get_single_train_dataloader(self, task_idx, n_samples=None, shuffle=True):
        dataset = self.get_split(task_idx, "train")

        if n_samples is not None:
            # sub-sample the dataset
            dataset = EquallySampledConcatDataset([dataset],
                                                  n_samples=n_samples,
                                                  shuffle=shuffle,
                                                  shuffle_dataset=False)

        if DEBUG:
            print(f"single train [{task_idx}: {len(dataset)}")

        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def get_train_dataloader(self, split, shuffle_datasets) -> DataLoader:

        datasets = []
        data_lengths = []
        for task_idx in self.tasks:
            d = self.get_split(task_idx, split)
            datasets.append(d)
            data_lengths.append(len(d))

        if self.sampling_method == "equal":
            dataset = EquallySampledConcatDataset(datasets,
                                                  n_samples=min(data_lengths),
                                                  shuffle=self.shuffle,
                                                  shuffle_dataset=shuffle_datasets)
        else:
            dataset = ConcatDataset(datasets)

        if DEBUG:
            print(f"dataset lengths: {data_lengths}")
            print(f"dataset len: {len(dataset)}")

        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
