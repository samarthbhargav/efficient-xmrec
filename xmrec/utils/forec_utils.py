"""
    Some handy functions for pytroch model training ...

    ## Sam: this is taken from the original repo
"""
import argparse
import os

import numpy as np
import torch
import random
import pytrec_eval


def compute_aggregated_measure(measure, values, include_std=False):
    # this is a clone of pytrec_eval.compute_aggregated_measure,
    # modified to also include the std devaiation
    if measure.startswith('num_'):
        assert not include_std
        agg_fun = np.sum
    elif measure.startswith('gm_'):
        assert not include_std

        def agg_fun(values):
            return np.exp(np.sum(values) / len(values))
    else:
        agg_fun = np.mean
    if not include_std:
        return agg_fun(values)
    else:
        return agg_fun(values), np.std(values)


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)
        return self.result

    def show(self, metrics, include_std=False):
        result = {}
        for metric in metrics:
            # res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            res = compute_aggregated_measure(metric, [user[metric] for user in self.result.values()],
                                             include_std=include_std)
            if include_std:
                result[metric] = {
                    "mean": res[0],
                    "std": res[1]
                }
            else:
                result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self, include_std=False):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys, include_std)


def get_evaluations_final(run_mf, test):
    metrics = {'recall_5', 'recall_10', 'recall_20', 'P_5', 'P_10', 'P_20', 'map_cut_10', 'ndcg_cut_10'}
    eval_obj = Evaluator(metrics)
    indiv_res = eval_obj.evaluate(run_mf, test)
    overall_res = eval_obj.show_all(include_std=True)
    return overall_res, indiv_res


def set_seed(args: argparse.Namespace, torch_deterministic: bool = True, torch_benchmark: bool = True) -> None:
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = torch_benchmark
    np.random.seed(seed)
    random.seed(seed)


# Checkpoints
def save_checkpoint(model, model_dir):
    print(f"saving checkpoint at {model_dir}")
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id, maml_bool=False, cuda=True):
    if cuda:
        map_location = lambda storage, loc: storage.cuda(device=device_id)
    else:
        map_location = "cpu"

    print(f"loading {model_dir}")
    state_dict = torch.load(model_dir,
                            map_location=map_location)  # ensure all storage are on gpu

    if maml_bool:
        for key in list(state_dict.keys()):
            new_key = key.replace('module.', '')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)


def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, network.parameters()),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),
                                     lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, network.parameters()),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def get_model_cid_dir(args, model_type, flip=False, checkpoint_dir="checkpoints"):
    """
    based on args and model type, this function generates idbank and checkpoint file dirs
    """
    # for single model experiments
    if hasattr(args, "experiment_type") and args.experiment_type == "single_model":
        if checkpoint_dir == "checkpoints":
            checkpoint_dir = "checkpoints_single_model"

        markets = "_".join(sorted(args.markets.split(",")))
        os.makedirs(checkpoint_dir, exist_ok=True)
        prefix = ""
        if hasattr(args, "market_aware") and args.market_aware:
            prefix = "mkt_aware_"

        model_dir = os.path.join(checkpoint_dir,
                                 f"{prefix}{markets}_{model_type}_{args.data_sampling_method}_{args.exp_name}.model")
        cid_dir = os.path.join(checkpoint_dir,
                               f'{prefix}{markets}_{model_type}_{args.data_sampling_method}_{args.exp_name}.pickle')
        return model_dir, cid_dir

    src_market = args.aug_src_market
    tgt_market = args.tgt_market
    if flip:
        src_market = args.tgt_market
        tgt_market = args.aug_src_market

    tmp_exp_name = f'{args.data_augment_method}_{args.data_sampling_method}'
    tmp_src_markets = src_market
    if args.data_augment_method == 'no_aug':
        # src_market = 'xx'
        tmp_exp_name = f'{args.data_augment_method}'
        tmp_src_markets = 'single'

    os.makedirs(checkpoint_dir, exist_ok=True)

    if hasattr(args, "market_aware") and args.market_aware:
        model_dir = os.path.join(checkpoint_dir,
                                 f"mkt_aware_{tgt_market}_{model_type}_{tmp_src_markets}_{tmp_exp_name}_{args.exp_name}.model")
        cid_dir = os.path.join(checkpoint_dir,
                               f'mkt_aware_{tgt_market}_{model_type}_{tmp_src_markets}_{tmp_exp_name}_{args.exp_name}.pickle')
    else:
        model_dir = os.path.join(checkpoint_dir,
                                 f"{tgt_market}_{model_type}_{tmp_src_markets}_{tmp_exp_name}_{args.exp_name}.model")
        cid_dir = os.path.join(checkpoint_dir,
                               f'{tgt_market}_{model_type}_{tmp_src_markets}_{tmp_exp_name}_{args.exp_name}.pickle')

    return model_dir, cid_dir


def get_model_config(model_type):
    gmf_config = {'alias': 'gmf',
                  'adam_lr': 0.005,  # 1e-3,
                  'latent_dim': 8,
                  'l2_regularization': 1e-07,  # 0, # 0.01
                  'embedding_user': None,
                  'embedding_item': None,
                  }

    mlp_config = {'alias': 'mlp',
                  'adam_lr': 0.01,  # 1e-3,
                  'latent_dim': 8,
                  'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                  'l2_regularization': 1e-07,  # 0.0000001,  # MLP model is sensitive to hyper params
                  'pretrain': True,
                  'embedding_user': None,
                  'embedding_item': None,
                  }

    neumf_config = {'alias': 'nmf',
                    'adam_lr': 0.01,  # 1e-3,
                    'latent_dim_mf': 8,
                    'latent_dim_mlp': 8,
                    'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                    'l2_regularization': 1e-07,  # 0.0000001, #0.01,
                    'pretrain': True,
                    'embedding_user': None,
                    'embedding_item': None,
                    }

    config = {
        'gmf': gmf_config,
        'mlp': mlp_config,
        'nmf': neumf_config}[model_type]

    return config


# conduct the testing on the model
def test_model(model, config, test_dataloader, test_qrel):
    print("evaluating model")
    model.eval()
    task_rec_all = []
    task_unq_users = set()
    mkt_idx = config.get("mkt_idx", None)
    for test_batch in test_dataloader:
        test_user_ids, test_item_ids, test_targets, test_markets = test_batch
        # _get_rankings function
        cur_users = [user.item() for user in test_user_ids]
        cur_items = [item.item() for item in test_item_ids]

        if mkt_idx:
            test_markets = torch.LongTensor([mkt_idx[m] for m in test_markets])
        if config['use_cuda'] is True:
            test_user_ids, test_item_ids, test_targets = test_user_ids.cuda(), test_item_ids.cuda(), test_targets.cuda()
            test_markets = test_markets.cuda()

        with torch.no_grad():
            batch_scores = model(test_user_ids, test_item_ids, market_indices=test_markets)
            if config['use_cuda'] is True:
                batch_scores = batch_scores.detach().cpu().numpy()
            else:
                batch_scores = batch_scores.detach().numpy()

        for index in range(len(test_user_ids)):
            task_rec_all.append((cur_users[index], cur_items[index], batch_scores[index][0].item()))

        task_unq_users = task_unq_users.union(set(cur_users))

    task_run_mf = get_run_mf(task_rec_all, task_unq_users)
    task_ov, task_ind = get_evaluations_final(task_run_mf, test_qrel)
    return task_ov, task_ind


def get_run_mf(rec_list, unq_users):
    ranking = {}
    for uid, iid, score in rec_list:
        if uid not in ranking:
            ranking[uid] = []
        ranking[uid].append((iid, score))

    run_mf = {}
    for uid in unq_users:
        r = ranking[uid]
        r.sort(key=lambda _: -_[1])
        cur_rank = {str(i): 2 + s for (i, s) in r}
        run_mf[str(uid)] = cur_rank

    return run_mf
