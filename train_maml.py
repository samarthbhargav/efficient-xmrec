import argparse
import time

import torch

import learn2learn as l2l
from tqdm import tqdm

from xmrec.data.data import MAMLTaskGenerator, MetaMarketDataloaders
from xmrec.models.model import NeuMF
import pandas as pd
import os
import json

import sys
import pickle

from xmrec.utils.forec_utils import test_model, set_seed, get_model_cid_dir, get_model_config, resume_checkpoint, \
    save_checkpoint, use_cuda


def create_arg_parser():
    parser = argparse.ArgumentParser('MAML_NeuMF_Engine')
    parser.add_argument("--experiment_type",
                        choices=["single_model", "pair"],
                        help="if 'single_model' aug_src_market argument is ignored and all markets are used",
                        default="pair")
    # Path Arguments
    parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
    # is kshots here, the default is 20, not 1024
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--num_neg', type=int, default=4, help='number of negatives to sample during training')
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')

    # output arguments
    parser.add_argument('--exp_name', help='name the experiment', type=str, required=True)
    parser.add_argument('--exp_output', help='output results .json file', type=str, required=True)

    # data arguments
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA2/proc_data')
    parser.add_argument('--tgt_market', help='specify target market', type=str, required=False,
                        default=None)  # de_Electronics
    parser.add_argument('--aug_src_market', help='which data to augment with', type=str, default='xx')  # us_Electronics

    # sampling_method:
    parser.add_argument('--data_sampling_method', help='in augmentation how to sample data for training', type=str,
                        default='concat', choices=['concat', 'equal'])

    # MAML arguments
    parser.add_argument('--fast_lr', type=float, default=0.1, help='meta-learning rate')
    # cold start setup
    parser.add_argument('--tgt_fraction', type=int, default=1, help='what fraction of data to use on target side')
    parser.add_argument('--src_fraction', type=int, default=1, help='what fraction of data to use from source side')

    # during Hyperparam search, we don't need to save the model
    parser.add_argument('--no_save', action="store_true", default=False,
                        help="disables persisting the final model if set")

    parser.add_argument('--market_aware', action="store_true", default=False,
                        help="learns market embeddings in addition to user/item embeddings")

    return parser


def fast_adapt(config, batch_adapt, batch_eval, learner, loss, adaptation_steps, mkt_idx):
    adapt_user_ids, adapt_item_ids, adapt_targets, adapt_markets = batch_adapt
    eval_user_ids, eval_item_ids, eval_targets, eval_markets = batch_eval

    adapt_markets = torch.LongTensor([mkt_idx[m] for m in adapt_markets])
    eval_markets = torch.LongTensor([mkt_idx[m] for m in eval_markets])

    if config['use_cuda'] is True:
        adapt_user_ids, adapt_item_ids, adapt_targets = adapt_user_ids.cuda(), adapt_item_ids.cuda(), adapt_targets.cuda()
        eval_user_ids, eval_item_ids, eval_targets = eval_user_ids.cuda(), eval_item_ids.cuda(), eval_targets.cuda()

        adapt_markets = adapt_markets.cuda()
        eval_markets = eval_markets.cuda()

    # Adapt the model
    for step in range(adaptation_steps):
        ratings_pred = learner(adapt_user_ids, adapt_item_ids, adapt_markets)
        train_error = loss(ratings_pred.view(-1), adapt_targets)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(eval_user_ids, eval_item_ids, eval_markets)
    valid_error = loss(predictions.view(-1), eval_targets)
    return valid_error


def run_batch(iterators, mkt, mkt_idx, maml, config, loss_func, adaptation_steps, train=True, learner=None):
    if learner is None:
        learner = maml.clone()
    adapt_batch = next(iterators[mkt])
    eval_batch = next(iterators[mkt])

    evaluation_error = fast_adapt(config,
                                  adapt_batch,
                                  eval_batch,
                                  learner,
                                  loss_func,
                                  adaptation_steps,
                                  mkt_idx)
    if train:
        evaluation_error.backward()

    return evaluation_error


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)

    if args.experiment_type == "pair":
        assert args.tgt_market is not None
        assert args.aug_src_market is not None
        markets = [args.tgt_market, args.aug_src_market]
    elif args.experiment_type == "single_model":
        args.markets = "de,jp,in,fr,ca,mx,uk"
        markets = args.markets.split(",")
    else:
        raise ValueError(args.experiment_type)

    args.data_augment_method = 'full_aug'
    args.model_selection = 'nmf'

    nmf_model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
    print(nmf_model_dir, cid_filename)
    with open(cid_filename, 'rb') as centralid_file:
        my_id_bank = pickle.load(centralid_file)

    ############
    ## All Market data
    ############
    task_gen_all = {}
    market_index = {}

    if args.experiment_type == "single_model":
        print("loading us data")
        # load the us market data
        us_data_dir = os.path.join(args.data_dir, f'us_10core.txt')
        us_ratings = pd.read_csv(us_data_dir, sep=" ")

        us_task_gen = MAMLTaskGenerator(us_ratings, "us", my_id_bank, item_thr=7)

        items_allowed = us_task_gen.item_pool_ids
        assert "us" not in markets
        task_gen_all[0] = us_task_gen
        market_index[0] = "us"

        print(f"data sampling method: {args.data_sampling_method}")
        for i, market in enumerate(markets, 1):
            print(f"loading {market}")
            ratings_path = os.path.join(args.data_dir, f'{market}_5core.txt')
            ratings = pd.read_csv(ratings_path, sep=" ")
            task_gen_all[i] = MAMLTaskGenerator(ratings, market, my_id_bank, item_thr=7)
            market_index[i] = market

    else:
        for mar_index, cur_market in enumerate(markets):
            cur_mkt_data_dir = os.path.join(args.data_dir, f'{cur_market}_5core.txt')
            if cur_market == 'us':
                cur_mkt_data_dir = os.path.join(args.data_dir, f'{cur_market}_10core.txt')
            print(f'loading {cur_mkt_data_dir}')
            cur_mkt_ratings = pd.read_csv(cur_mkt_data_dir, sep=' ')

            cur_mkt_fraction = args.src_fraction if mar_index >= 1 else args.tgt_fraction

            task_generator = MAMLTaskGenerator(
                ratings=cur_mkt_ratings,
                market=cur_market,
                id_index_bank=my_id_bank,
                item_thr=7,
                sample_df=cur_mkt_fraction)

            task_gen_all[mar_index] = task_generator
            market_index[mar_index] = cur_market

    print('loaded all data!')

    sys.stdout.flush()

    ############
    ## Dataset Concatenation
    ############
    sampling_method = args.data_sampling_method  # 'concat'  'equal'

    dataloaders = MetaMarketDataloaders(tasks=task_gen_all,
                                        sampling_method=sampling_method,
                                        num_train_negatives=args.num_neg,
                                        batch_size=args.batch_size,
                                        shuffle=True)
    ############
    ## Model Prepare
    ############
    all_model_selection = ['nmf']

    results = {}

    for cur_model_selection in all_model_selection:
        sys.stdout.flush()
        args.model_selection = cur_model_selection
        config = get_model_config(args.model_selection)
        config['batch_size'] = args.batch_size
        config['optimizer'] = 'adam'
        config['use_cuda'] = args.cuda
        config['device_id'] = 0
        config['save_trained'] = False if args.no_save else True
        config['load_pretrained'] = True
        config['num_users'] = int(my_id_bank.last_user_index + 1)
        config['num_items'] = int(my_id_bank.last_item_index + 1)
        config['num_markets'] = len(markets)
        config['market_aware'] = args.market_aware
        config["mkt_idx"] = {m: i for (i, m) in market_index.items()}

        model = NeuMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            model.cuda()
        resume_checkpoint(model, model_dir=nmf_model_dir, device_id=config['device_id'], cuda=config["use_cuda"])
        print(model)
        sys.stdout.flush()

        fast_lr = args.fast_lr  # =0.5
        # meta_batch_size = train_dataloader.num_tasks  # 32
        adaptation_steps = 1
        test_adaptation_steps = 1  # how many times adapt the model for testing time

        maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        opt = torch.optim.Adam(maml.parameters(), lr=config['adam_lr'], weight_decay=config['l2_regularization'])
        loss_func = torch.nn.BCELoss()

        ############
        ## Train
        ############

        # do a dummy run to figure out how many batches are there for each
        # market
        samples = {}
        for mkt_idx, mkt in market_index.items():
            samples[mkt_idx] = len(dataloaders.get_split(mkt_idx, "train"))

        # the original code oversamples the smaller markets
        # but since we want to do epochs based on target markets
        # we will use min instead of max
        n_samples = min(samples.values())

        start_time = time.time()
        for epoch in tqdm(range(args.num_epoch), desc="Train"):
            sys.stdout.flush()

            # train_dl_dict = {}
            iterators = {}
            valid_iterators = {}
            n_iterations = {}
            for mkt_idx, mkt in market_index.items():
                train_dl = dataloaders.get_single_train_dataloader(mkt_idx,
                                                                   n_samples=n_samples,
                                                                   shuffle=True)
                iterators[mkt_idx] = iter(train_dl)
                valid_iterators[mkt_idx] = iter(dataloaders.get_valid_dataloader(mkt_idx, "valid"))
                n_iterations[mkt_idx] = len(train_dl)

            # since 2 batches are extracted at a time
            n_iterations = int(min(n_iterations.values()) / 2)
            for iter_num in tqdm(range(n_iterations), desc=f"iter epoch {epoch}", leave=False):
                opt.zero_grad()
                meta_train_loss = 0.0
                meta_valid_loss = 0.0
                for mkt in market_index:
                    evaluation_error = run_batch(iterators,
                                                 mkt,
                                                 config["mkt_idx"],
                                                 maml,
                                                 config,
                                                 loss_func,
                                                 adaptation_steps,
                                                 train=True)

                    meta_train_loss += evaluation_error.item()

                    evaluation_error_val = run_batch(valid_iterators,
                                                     mkt,
                                                     config["mkt_idx"],
                                                     maml,
                                                     config,
                                                     loss_func,
                                                     adaptation_steps,
                                                     train=False)

                    meta_valid_loss += evaluation_error_val.item()

            for p in maml.parameters():
                p.grad.data.mul_(1.0 / len(iterators))
            opt.step()

        train_time = time.time() - start_time
        ############
        ## TEST
        ############
        cur_model_results = {}

        test_iterators = {}
        test_qrel = {}

        valid_iterators = {}
        valid_dataloaders = {}
        test_dataloaders = {}
        valid_qrel = {}
        n_iterations = {}
        for mkt_idx, mkt in market_index.items():
            valid_dataloaders[mkt_idx] = dataloaders.get_valid_dataloader(mkt_idx, "valid")
            valid_iterators[mkt_idx] = iter(valid_dataloaders[mkt_idx])
            valid_qrel[mkt_idx] = task_gen_all[mkt_idx].get_validation_qrel("valid")

            test_dataloaders[mkt_idx] = dataloaders.get_valid_dataloader(mkt_idx, "test")
            test_iterators[mkt_idx] = iter(test_dataloaders[mkt_idx])
            test_qrel[mkt_idx] = task_gen_all[mkt_idx].get_validation_qrel("test")


        for cur_market in markets:
            market_index_inv = {m: i for (i, m) in market_index.items()}
            mar_index = market_index_inv[cur_market]
            # validation data
            learner = maml.clone()

            for test_adapt_step in range(test_adaptation_steps):
                evaluation_error = run_batch(valid_iterators,
                                             mar_index,
                                             config["mkt_idx"],
                                             maml,
                                             config,
                                             loss_func,
                                             adaptation_steps,
                                             train=False,
                                             learner=learner)

                print(f'test eval on {cur_market} adaptation step: {test_adapt_step}')
                valid_ov, valid_ind = test_model(learner, config, valid_dataloaders[mar_index], valid_qrel[mar_index])
                cur_ndcg = valid_ov['ndcg_cut_10']
                cur_recall = valid_ov['recall_10']
                print(
                    f'[pytrec_based] Market: {cur_market} step{test_adapt_step} tgt_valid: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall}')

                cur_model_results[f'valid_{cur_market}_step{test_adapt_step}'] = {
                    'agg': valid_ov,
                    'ind': valid_ind,
                }

                # test data
                test_ov, test_ind = test_model(learner, config, test_dataloaders[mar_index], test_qrel[mar_index])
                cur_ndcg = test_ov['ndcg_cut_10']
                cur_recall = test_ov['recall_10']
                print(
                    f'[pytrec_based] Market: {cur_market} step{test_adapt_step} tgt_test: \t NDCG@10: {cur_ndcg} \t R@10: {cur_recall} \n\n')

                cur_model_results[f'test_{cur_market}_step{test_adapt_step}'] = {
                    'agg': test_ov,
                    'ind': test_ind,
                }

        cur_model_results["train_time"] = train_time
        results[args.model_selection] = cur_model_results

        ############
        ## SAVE the model
        ############
        if config['save_trained']:
            # model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
            maml_nmf_output_dir = nmf_model_dir.replace('/', f'/maml{args.batch_size}_')
            save_checkpoint(maml, maml_nmf_output_dir)

    os.makedirs(os.path.dirname(args.exp_output), exist_ok=True)
    # writing the results into a file
    results['args'] = vars(args)
    with open(args.exp_output, 'w') as outfile:
        json.dump(results, outfile)
    print('Experiment finished success!')
