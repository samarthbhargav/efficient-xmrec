import argparse
import pandas as pd

from xmrec.data.data import MAMLTaskGenerator, MetaMarketDataloaders
from xmrec.models.model import NeuMF_MH

import os
import json

import sys
import pickle

from xmrec.utils.forec_utils import set_seed, get_model_cid_dir, get_model_config, resume_checkpoint, save_checkpoint

from train_base import train_and_test_model


def create_arg_parser():
    parser = argparse.ArgumentParser('FOREC_NeuMF_Engine')

    parser.add_argument("--experiment_type",
                        choices=["single_model", "pair"],
                        help="if 'single_model' aug_src_market argument is ignored and all markets are used",
                        default="pair")

    # Path Arguments
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')

    parser.add_argument('--maml_shots', type=int, default=20, help='k shots for MAML')

    parser.add_argument('--num_neg', type=int, default=4, help='number of negatives to sample during training')
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')

    # output arguments 
    parser.add_argument('--exp_name', help='name the experiment', type=str, required=True)
    parser.add_argument('--exp_output', help='output results .json file', type=str, required=True)

    # data arguments 
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA2/proc_data')
    parser.add_argument('--tgt_market', help='specify target market', type=str, required=False, default=None)
    parser.add_argument('--aug_src_market', help='which data to augment with', type=str, default='xx')

    parser.add_argument('--data_sampling_method', help='in augmentation how to sample data for training', type=str,
                        default='concat')
    parser.add_argument('--tgt_fraction', type=int, default=1, help='what fraction of data to use on target side')
    parser.add_argument('--src_fraction', type=int, default=1, help='what fraction of data to use from source side')

    # during Hyperparam search, we don't need to save the model
    parser.add_argument('--no_save', action="store_true", default=False,
                        help="disables persisting the final model if set")

    return parser


"""
The main module that takes the model and dataloaders for training and testing on specific target market 
"""


def freeze_model(model, allowed_fc_layers=None):
    if allowed_fc_layers is None:
        allowed_fc_layers = []
    # freeze all the parameters 
    for param in model.parameters():
        param.requires_grad = False
    #     print(param.shape, param.requires_grad)

    for allowed_fc_layer in allowed_fc_layers:
        model.fc_layers[allowed_fc_layer].weight.requires_grad = True
        model.fc_layers[allowed_fc_layer].bias.requires_grad = True

    model.affine_output.weight.requires_grad = True
    model.affine_output.bias.requires_grad = True
    model.logistic.requires_grad = True
    return model


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)

    if args.experiment_type == "pair":
        assert args.tgt_market is not None
        assert args.aug_src_market is not None
        cur_tgt_markets = [args.tgt_market, args.aug_src_market]
    elif args.experiment_type == "single_model":
        args.markets = "de,jp,in,fr,ca,mx,uk"
        cur_tgt_markets = args.markets.split(",")
    else:
        raise ValueError(args.experiment_type)

    ############
    ## load id bank
    ############
    args.data_augment_method = 'full_aug'  # 'concat'
    args.model_selection = 'nmf'

    nmf_model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
    with open(cid_filename, 'rb') as centralid_file:
        my_id_bank = pickle.load(centralid_file)

    # for maml model loading:
    maml_nmf_load = True
    maml_shots = args.maml_shots
    nmf_model_dir = nmf_model_dir.replace('/', f'/maml{maml_shots}_')

    ############
    ## Train for every target market (just for demonstration purpose, in fact you only need the target market here)
    ############
    results = {}
    MH_spec = {'mh_layers': [16, 32, 16], 'adam_lr': 0.01, 'l2_regularization': 0.001}

    for cur_tgt_market in cur_tgt_markets:
        cur_mark_fraction = args.tgt_fraction
        if cur_tgt_market == args.aug_src_market:
            cur_mark_fraction = args.src_fraction

        cur_mkt_data_dir = os.path.join(args.data_dir, f'{cur_tgt_market}_5core.txt')
        if cur_tgt_market == 'us':
            cur_mkt_data_dir = os.path.join(args.data_dir, f'{cur_tgt_market}_10core.txt')
        print(f'loading {cur_mkt_data_dir}')

        cur_mkt_ratings = pd.read_csv(cur_mkt_data_dir, sep=' ')

        # tgt_task_generator = TaskGenerator(cur_mkt_ratings, my_id_bank, item_thr=7, sample_df=cur_mark_fraction)
        # task_gen_all = {
        #     0: tgt_task_generator,
        # }
        # train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=4, meta_split='train')
        # train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True,
        #                                          num_workers=0)

        tgt_task_generator = MAMLTaskGenerator(
            ratings=cur_mkt_ratings,
            id_index_bank=my_id_bank,
            market=cur_tgt_market,
            item_thr=7,
            sample_df=cur_mark_fraction)

        task_gen_all = {
            0: tgt_task_generator
        }

        train_dataloaders = MetaMarketDataloaders(tasks=task_gen_all,
                                                  sampling_method="no_aug",
                                                  num_train_negatives=args.num_neg,
                                                  batch_size=args.batch_size)

        print('loaded target data!')
        sys.stdout.flush()

        print('preparing test/valid data...')

        tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(0,
                                                                                     sample_batch_size=args.batch_size,
                                                                                     shuffle=False, num_workers=0,
                                                                                     split='valid')
        tgt_valid_qrel = tgt_task_generator.get_validation_qrel(split='valid')

        tgt_test_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(0,
                                                                                    sample_batch_size=args.batch_size,
                                                                                    shuffle=False, num_workers=0,
                                                                                    split='test')
        tgt_test_qrel = tgt_task_generator.get_validation_qrel(split='test')

        # load the pretrained nmf model
        nmf_config = get_model_config(args.model_selection)
        nmf_config['num_users'] = int(my_id_bank.last_user_index + 1)
        nmf_config['num_items'] = int(my_id_bank.last_item_index + 1)
        nmf_config['batch_size'] = args.batch_size
        nmf_config['optimizer'] = 'adam'
        nmf_config['use_cuda'] = args.cuda
        nmf_config['device_id'] = 0
        nmf_config['save_trained'] = False if args.no_save else True
        nmf_config["mkt_idx"] = {m: i for (i, m) in enumerate([cur_tgt_market])}

        for conf_key, conf_val in MH_spec.items():
            nmf_config[conf_key] = conf_val

        model = NeuMF_MH(nmf_config)
        if nmf_config['use_cuda'] is True:
            model.cuda()

        if maml_nmf_load:
            resume_checkpoint(model, model_dir=nmf_model_dir, device_id=nmf_config['device_id'], maml_bool=True,
                              cuda=nmf_config["use_cuda"])
        else:
            resume_checkpoint(model, model_dir=nmf_model_dir, device_id=nmf_config['device_id'],
                              cuda=nmf_config["use_cuda"])
        sys.stdout.flush()

        # freeze desired layers
        args.unfreeze_from = -3
        if args.unfreeze_from != 0:
            cur_unfreeze_from = int(args.unfreeze_from)
            allowed_fc_layers = [idx for idx in range(cur_unfreeze_from, 0)]
            model = freeze_model(model, allowed_fc_layers=allowed_fc_layers)

        for allowed_mh_layer in range(len(model.mh_layers)):
            model.mh_layers[allowed_mh_layer].weight.requires_grad = True
            model.mh_layers[allowed_mh_layer].bias.requires_grad = True

        print('model shape and freeze status: \n')
        for param in model.parameters():
            print(param.shape, param.requires_grad)

        model, cur_model_results = train_and_test_model(args, nmf_config, model, train_dataloaders,
                                                        tgt_valid_dataloader,
                                                        tgt_valid_qrel, tgt_test_dataloader, tgt_test_qrel)

        results[cur_tgt_market] = {}
        for k, v in cur_model_results.items():
            results[cur_tgt_market][k] = v

        ############
        ## SAVE the model
        ############
        if nmf_config['save_trained']:
            # model_dir, cid_filename = get_model_cid_dir(args, args.model_selection)
            forec_model_output_dir = nmf_model_dir.replace('/', f'/forec{args.batch_size}_{cur_tgt_market}_')
            save_checkpoint(model, forec_model_output_dir)

    os.makedirs(os.path.dirname(args.exp_output), exist_ok=True)
    # writing the results into a file
    results['args'] = vars(args)
    with open(args.exp_output, 'w') as outfile:
        json.dump(results, outfile)

    print('Experiment finished success!')
