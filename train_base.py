"""
This script trains all GMF, MLP, and NMF baselines for a single market
Provides three options for the use of another source market:
  1. 'no_aug'  : only use the target market train data, hence single market training (the src market will set to 'xx')
  2. 'full_aug': fully uses the source market data for training
  3. 'sel_aug' : only use portion of source market data covering target market's items
  
For data sampling:
  a. 'equal'   : equally sample data from both source and target markets, providing a balanced training
  b. 'concate' : first concatenate the source and target training data, treat that a single training data
"""

import time
import pandas as pd

import argparse

import torch
from tqdm.autonotebook import tqdm

from xmrec.data.data import CentralIDBank, MAMLTaskGenerator, MetaMarketDataloaders
from xmrec.models.model import GMF, MLP, NeuMF

import os
import json
import sys
import pickle

from xmrec.utils.forec_utils import use_optimizer, test_model, set_seed, get_model_config, get_model_cid_dir, \
    save_checkpoint, \
    use_cuda


def create_arg_parser():
    parser = argparse.ArgumentParser('NeuMF_Engine')
    # Path Arguments
    parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_neg', type=int, default=4, help='number of negatives to sample during training')
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')

    # output arguments 
    parser.add_argument('--exp_name', help='name the experiment', type=str, required=True)
    parser.add_argument('--exp_output', help='output results .json file', type=str, required=True)

    # data arguments 
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA2/proc_data/')
    parser.add_argument('--tgt_market', help='specify target market', type=str, required=True)  # de_Electronics
    parser.add_argument('--aug_src_market', help='which data to augment with', type=str, default='xx')  # us_Electronics

    # augmentation approaches
    # aug_method:
    parser.add_argument('--data_augment_method', required=True, help='how to augment data to target market', type=str,
                        choices=['no_aug', 'full_aug', 'sel_aug'])
    # sampling_method: 'concat'  'equal'
    parser.add_argument('--data_sampling_method', help='in augmentation how to sample data for training', type=str,
                        choices=['concat', 'equal'])

    # MODEL selection
    parser.add_argument('--model_selection', help='which nn model to train with', type=str, default="all",
                        choices=["gmf", "mlp", "nmf", "all"])

    # during Hyperparam search, we don't need to save the model
    parser.add_argument('--no_save', action="store_true", default=False,
                        help="disables persisting the final model if set")

    parser.add_argument('--market_aware', action="store_true", default=False,
                        help="learns market embeddings in addition to user/item embeddings")

    # cold start setup
    parser.add_argument('--tgt_fraction', type=int, default=1, help='what fraction of data to use on target side')
    parser.add_argument('--src_fraction', type=int, default=1, help='what fraction of data to use from source side')

    return parser


def train_and_test_model(args, config, model, dataloaders, valid_dataloader, valid_qrel, test_dataloader,
                         test_qrel, cur_tgt=""):
    opt = use_optimizer(model, config)
    loss_func = torch.nn.BCELoss()

    ############
    ## Train
    ############
    best_ndcg = 0.0
    best_eval_res = {}
    all_eval_res = {}
    start_time = time.time()
    mkt_idx = config["mkt_idx"]
    for epoch in tqdm(range(args.num_epoch), desc="Train"):
        model.train()
        total_loss = 0

        train_dl = dataloaders.get_train_dataloader("train", shuffle_datasets=True)
        for train_user_ids, train_item_ids, train_targets, train_markets in tqdm(train_dl, desc=f"Epoch {epoch}",
                                                                                 leave=False):
            train_markets = torch.LongTensor([mkt_idx[m] for m in train_markets])

            if config['use_cuda'] is True:
                train_user_ids = train_user_ids.cuda()
                train_item_ids = train_item_ids.cuda()
                train_targets = train_targets.cuda()
                train_markets = train_markets.cuda()

            opt.zero_grad()
            ratings_pred = model(train_user_ids, train_item_ids, market_indices=train_markets)
            loss = loss_func(ratings_pred.view(-1), train_targets)
            loss.backward()
            opt.step()
            total_loss += loss.item()

    all_eval_res["train_time"] = time.time() - start_time
    ############
    ## TEST
    ############
    # if args.model_selection=='nmf':
    print("evaluation")
    valid_ov, valid_ind = test_model(model, config, valid_dataloader, valid_qrel)
    cur_ndcg = valid_ov['ndcg_cut_10']
    cur_recall = valid_ov['recall_10']
    print(f'[pytrec_based] {cur_tgt} tgt_valid: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall}')

    all_eval_res[f'valid'] = {
        'agg': valid_ov,
        'ind': valid_ind,
    }

    test_ov, test_ind = test_model(model, config, test_dataloader, test_qrel)
    cur_ndcg = test_ov['ndcg_cut_10']
    cur_recall = test_ov['recall_10']
    print(f'[pytrec_based] {cur_tgt} tgt_test: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall} \n\n')

    all_eval_res[f'test'] = {
        'agg': test_ov,
        'ind': test_ind,
    }

    print(f"evaluation complete: {(time.time() - start_time) / 60:.2f} minutes since start")

    return model, all_eval_res


def get_data(args):
    id_bank = CentralIDBank()
    tgt_data_dir = os.path.join(args.data_dir, f'{args.tgt_market}_5core.txt')
    print(f'loading {tgt_data_dir}')
    tgt_ratings = pd.read_csv(tgt_data_dir, sep=' ')

    tgt_task_generator = MAMLTaskGenerator(tgt_ratings, args.tgt_market, id_bank, item_thr=7,
                                           sample_df=args.tgt_fraction)

    aug_method = args.data_augment_method
    if args.aug_src_market == 'us':
        src_data_dir = os.path.join(args.data_dir, f'{args.aug_src_market}_10core.txt')
    else:
        src_data_dir = os.path.join(args.data_dir, f'{args.aug_src_market}_5core.txt')

    if aug_method == 'no_aug':
        src_task_generator = None
        args.aug_src_market = 'xx'
    elif aug_method == 'full_aug':
        print(f'loading {src_data_dir}')
        src_ratings = pd.read_csv(src_data_dir, sep=' ')
        src_task_generator = MAMLTaskGenerator(src_ratings, args.aug_src_market, id_bank, item_thr=7,
                                               sample_df=args.src_fraction)
    elif aug_method == 'sel_aug':
        print(f'loading {src_data_dir} with limiting to target data item pool...')
        src_ratings = pd.read_csv(src_data_dir, sep=' ')
        aug_items_allowed = tgt_task_generator.item_pool_ids
        src_task_generator = MAMLTaskGenerator(src_ratings, args.aug_src_market, id_bank, item_thr=7,
                                               items_allow=aug_items_allowed)
    else:
        raise ValueError(aug_method)

    sampling_method = args.data_sampling_method  # 'concat'  'equal'

    # 0. only use the target market train data
    if aug_method == "no_aug":
        task_gen_all = {
            0: tgt_task_generator,
        }
        markets = [args.tgt_market]
    elif sampling_method in {"equal", "concat"}:
        task_gen_all = {
            0: tgt_task_generator,
            1: src_task_generator
        }
        markets = [args.tgt_market, args.aug_src_market]
    else:
        raise ValueError(sampling_method)

    dataloaders = MetaMarketDataloaders(task_gen_all,
                                        sampling_method=sampling_method,
                                        num_train_negatives=args.num_neg,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=0)

    return id_bank, markets, tgt_task_generator, dataloaders


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)

    id_bank, markets, tgt_task_generator, dataloaders = get_data(args)

    print('preparing valid data...')

    tgt_valid_dataloader = dataloaders.get_valid_dataloader(0, "valid")

    print("preparing qrel")
    tgt_valid_qrel = tgt_task_generator.get_validation_qrel(split='valid')

    print("preparing test data")
    tgt_test_dataloader = dataloaders.get_valid_dataloader(0, "test")
    print("preparing qrel")
    tgt_test_qrel = tgt_task_generator.get_validation_qrel(split='test')

    ############
    ## Model Prepare
    ############

    if args.model_selection == "all":
        models = ['gmf', 'mlp', 'nmf']
    else:
        models = [args.model_selection]

    results = {}
    for cur_model_selection in models:
        sys.stdout.flush()
        args.model_selection = cur_model_selection
        config = get_model_config(args.model_selection)
        config['batch_size'] = args.batch_size
        config['optimizer'] = 'adam'
        config['use_cuda'] = args.cuda
        config['market_aware'] = args.market_aware
        config['device_id'] = 0
        config['save_trained'] = False if args.no_save else True
        config['load_pretrained'] = True
        config['num_users'] = int(id_bank.last_user_index + 1)
        config['num_items'] = int(id_bank.last_item_index + 1)
        config['num_markets'] = len(markets)
        config['mkt_idx'] = {m: i for (i, m) in enumerate(sorted(markets))}

        if args.model_selection == 'gmf':
            print('model is GMF!')
            model = GMF(config)
        elif args.model_selection == 'nmf':
            print('model is NeuMF!')
            model = NeuMF(config)
            if config['load_pretrained']:
                print('loading pretrained gmf and mlp...')
                model.load_pretrain_weights(args)
        else:  # default is MLP
            print('model is MLP!')
            model = MLP(config)
            if config['load_pretrained']:
                print('loading pretrained gmf...')
                model.load_pretrain_weights(args)

        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            model.cuda()
        print(model)
        sys.stdout.flush()
        model, cur_model_results = train_and_test_model(args, config, model, dataloaders, tgt_valid_dataloader,
                                                        tgt_valid_qrel, tgt_test_dataloader, tgt_test_qrel)

        # if args.model_selection=='nmf':
        results[args.model_selection] = cur_model_results

        ############
        ## SAVE the model and idbank
        ############
        if config['save_trained']:
            model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
            save_checkpoint(model, model_dir)
            with open(cid_filename, 'wb') as centralid_file:
                pickle.dump(id_bank, centralid_file)
    if len(os.path.dirname(args.exp_output)) > 0:
        os.makedirs(os.path.dirname(args.exp_output), exist_ok=True)

    # writing the results into a file
    results['args'] = vars(args)
    with open(args.exp_output, 'w') as outfile:
        json.dump(results, outfile)

    print('Experiment finished success!')
