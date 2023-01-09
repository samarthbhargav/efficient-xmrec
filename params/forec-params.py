import argparse
import copy
import json
import os

from params import prod_dict, make_cmd

CMD = "python "
if __name__ == '__main__':
    parser = argparse.ArgumentParser("forec-params",
                                     description="Utility for generating scripts to reproduce the original FOREC paper")
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--data_aug", required=True, type=str, help="data augment method",
                        choices=['no_aug', 'full_aug', 'sel_aug'])
    parser.add_argument("--sampling", required=True, type=str, help="sampling method used",
                        choices=['concat', 'equal'])
    parser.add_argument("--data_path", required=True, type=str, help="path to data")
    parser.add_argument("--out_path", required=True, type=str, help="path to output of experiments")

    parser.add_argument("--tgt", required=True, type=str, help="csv of target markets")
    parser.add_argument("--src", required=True, type=str, help="csv of src markets")

    parser.add_argument("--tgt_fraction", required=False, default=1, type=int,
                        help="fraction of the tgt data to use (note: this is the denominator, for 50% (1/2) enter 2)")
    parser.add_argument("--src_fraction", required=False, default=1, type=int,
                        help="fraction of the src data to use (note: this is the denominator, for 50% (1/2) enter 2)")

    parser.add_argument("--hyperparams_file", required=False, default=None, type=str,
                        help="optional file, if given will enumerate the hyperapameters provided")

    parser.add_argument("--num_epochs", required=False, default=25, type=int,
                        help="number of epochs to train")
    parser.add_argument("--batch_size", required=False, default=1024, type=int,
                        help="batch size used during training")

    parser.add_argument("--maml_shots", required=False, default=20, type=int,
                        help="number of epochs to train")
    parser.add_argument("--maml_fast_lr_tune", required=False, default="0.1", type=str,
                        help="number of epochs to train")

    parser.add_argument("--job_config", help="(json) location of job config", required=True)

    parser.add_argument("--dest", help="location to dump script + hparams", required=True)
    parser.add_argument('--market_aware', action="store_true", default=False,
                        help="makes models market_aware (only available for few models)")

    parser.add_argument("--job_template", help="location of job template", default="./job_template.sh")
    parser.add_argument("--device", help="device to run experiments on", default="cuda")

    args = parser.parse_args()

    # ##########
    # General Args
    # ##########
    exp_name = args.exp_name
    data_augment_method = args.data_aug
    sampling_method = args.sampling

    exp_name_com = f'{exp_name}_{sampling_method}'

    cur_data_path = args.data_path
    exp_output_dir = args.out_path

    print(f'\n- experiment name: {exp_name_com}')
    print(f'\t - data augmentation methods: {data_augment_method}')
    print(f'\t - data sampling method: {sampling_method}')
    print(f'\t - reading data: {cur_data_path}')
    print(f'\t - writing evaluations: {exp_output_dir}')

    # ##########
    # Market selection
    target_markets = args.tgt.split(",")
    assert len(target_markets) > 1
    source_markets = args.src.split(",")

    print(f'-Working on below market pairs (target, augmenting with market):')
    all_poss_pairs = []
    for target_market in target_markets:
        for source_market in source_markets:
            if target_market == source_market:
                continue
            if data_augment_method == 'no_aug':
                source_market = 'xx'
            all_poss_pairs.append((target_market, source_market))
            print(f'\t--> ({target_market}, {source_market})')
    all_poss_pairs = list(set(all_poss_pairs))

    # ##########
    # Training Data fractions to use from each target and source
    # 1 means full data, and 2 means 1/2 of the training data to sample
    # ##########
    tgt_fractions = [args.tgt_fraction]
    src_fractions = [args.src_fraction]  # 2, 3, 4, 5, 10

    fractions = []
    print('\n-Sampling below training data fractions:')
    for tgt_fraction in tgt_fractions:
        for src_fraction in src_fractions:
            fractions.append((src_fraction, tgt_fraction))
            print(f'\t--> ({src_fraction}, {tgt_fraction})')

    command_dict = {}

    hyperparams = {
        "num_epoch": [args.num_epochs],
        "batch_size": [args.batch_size]
    }

    # load hyperparameters to search over
    if args.hyperparams_file is not None:
        with open(args.hyperparams_file) as reader:
            search_space = json.load(reader)
        assert search_space["type"] == "hyperparams"
        del search_space["type"]
        is_hyperparam_search = True
        hyperparams.update(search_space)
    else:
        is_hyperparam_search = False

    for tgt_market, src_market in all_poss_pairs:
        for tgt_frac, src_fra in fractions:
            for i, model_config in enumerate(prod_dict(hyperparams)):

                if is_hyperparam_search:
                    cur_exp_name = f'{exp_name_com}_{tgt_market}_{src_market}_{data_augment_method}_ftgt{tgt_frac}_fsrc{src_fra}_hs{i}'
                    no_save = True
                else:
                    cur_exp_name = f'{exp_name_com}_{tgt_market}_{src_market}_{data_augment_method}_ftgt{tgt_frac}_fsrc{src_fra}'
                    no_save = False

                if args.market_aware:
                    cur_exp_name += "_market_aware"

                pre_set_args = {
                    "data_dir": cur_data_path,
                    "tgt_market": tgt_market,
                    "aug_src_market": src_market,
                    "exp_name": cur_exp_name,
                    "cuda": True,
                    "data_augment_method": data_augment_method,
                    "data_sampling_method": sampling_method,
                    "tgt_fraction": tgt_frac,
                    "src_fraction": src_fra,
                    "no_save": no_save
                }

                ### BASE Models ####
                base_args = copy.deepcopy(pre_set_args)
                base_args.update(copy.deepcopy(model_config))
                base_args["exp_output"] = f'{exp_output_dir}base-{cur_exp_name}.json'
                if args.market_aware:
                    base_args["market_aware"] = True

                final_cmd = " ".join(["train_base.py", make_cmd(base_args)])

                cur_cmd_dict = {'base': final_cmd}

                # if a hyperparameter search is being done,
                # then only the base models have to be trained
                # subsequent training of MAML/FOREC is done on
                # the 'best' model based on validation set performance
                if is_hyperparam_search or data_augment_method == 'no_aug':
                    command_dict[cur_exp_name] = cur_cmd_dict

                    continue

                #### MAML ####
                maml_args = copy.deepcopy(pre_set_args)
                # this is full_aug, so no need to set
                del maml_args["data_augment_method"]
                # for MAML, batch_size == shots
                maml_args["batch_size"] = args.maml_shots  # 512, 200, 100, 50, 20
                maml_args["fast_lr"] = args.maml_fast_lr_tune
                maml_args["exp_output"] = f'{exp_output_dir}maml-{cur_exp_name}_shots{args.maml_shots}.json'
                cur_cmd_dict['maml'] = " ".join(["train_maml.py", make_cmd(maml_args)])

                #### FOREC ####
                forec_args = copy.deepcopy(pre_set_args)
                del forec_args["data_augment_method"]
                forec_args["exp_output"] = f'{exp_output_dir}forec-{cur_exp_name}_shots{args.maml_shots}.json'
                forec_args["batch_size"] = args.maml_shots  # 512, 200, 100, 50, 20
                cur_cmd_dict['forec'] = " ".join(['train_forec.py', make_cmd(forec_args)])

                command_dict[cur_exp_name] = cur_cmd_dict

    print(f'Generated {len(command_dict)} experiments:')
    for k, v in command_dict.items():
        print(f'{k}')
        print(f'\t{list(v.keys())}')

    with open(args.job_config, "r") as reader:
        job_config = json.load(reader)

    with open(args.job_template, "r") as reader:
        job_template = ""
        for line in reader:
            job_template += line

    job_files = []

    for model_sel in ["base", "maml", "forec"]:
        job_file = f"{args.dest}_{model_sel}.job"
        hparams_file = f"{args.dest}_{model_sel}.params"

        if os.path.exists(job_file) or os.path.exists(hparams_file):
            raise ValueError(f"{job_file} OR {hparams_file} already exists!")

        job_device = job_config["device"]

        n_jobs = 0

        # seeds = [random.randint(0, 10000) for _ in range(args.n_seeds)]
        with open(hparams_file, "w") as writer:
            for cur_exp_name, v in command_dict.items():
                if model_sel in v:
                    writer.write(v[model_sel] + '\n')
                    n_jobs += 1

        print(f"N Jobs: {n_jobs} for {model_sel}")
        if n_jobs == 0:
            os.remove(hparams_file)
            continue

        final_cmd = f"{CMD} $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)"

        with open(job_file, "w") as writer:
            job = job_template.format(n_jobs=n_jobs, hyperparams_file=hparams_file, cmd=final_cmd, **job_config)
            writer.write(job)

        job_files.append(job_file)

    print("Execution order: ")
    with open(f"{args.dest}.sh", "w") as writer:
        for job_file in job_files:
            cmd = f"sbatch {job_file}"
            writer.write(cmd + "\n")
            writer.write("python block_srun.py\n")

    print(f"Execute: {args.dest}")
