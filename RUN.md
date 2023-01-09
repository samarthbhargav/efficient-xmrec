# Reproducing the results

Note: This requires SLURM to run on a cluster, but can be adapted to 
run using other configurations. 

Ensure that you edit the `job_config/job_config.json`,  files!

## Pairwise Model Experiments

```
cd params/
# Single-Market Baselines
python forec-params.py --exp_name forec_single --data_aug no_aug --sampling equal --data_path DATA2/proc_data --out_path forec_eval_single/ --tgt jp,in,de,fr,ca,mx,uk --src xx --job_config ./job_config/job_config.json --dest forec_single

# Pairwise-Baselines
python forec-params.py --exp_name forec_all --data_aug full_aug --sampling equal --data_path DATA2/proc_data --out_path forec_eval_all/ --tgt jp,in,de,fr,ca,mx,uk --src jp,in,de,fr,ca,mx,uk,us --job_config ./job_config/job_config.json --dest forec_all

# Market Aware Models
python forec-params.py --market_aware --exp_name forec_all_market_aware --data_aug full_aug --sampling equal --data_path DATA2/proc_data --out_path forec_eval_all_market_aware/ --tgt jp,in,de,fr,ca,mx,uk --src jp,in,de,fr,ca,mx,uk,us --job_config ./job_config/job_config.json --dest forec_all_market_aware

mv forec_single_base.params forec_single_base.job ../
rm forec_all_market_aware_forec.* forec_all_market_aware_maml.*
mv forec_all_* ../ 

cd ../

# Execute one after the other to avoid launching 
# too many jobs!
# Single Market Job
sbatch forec_single_base.job
# Base Market-unaware models
sbatch forec_all_base.job
# MAML models, needs previous
sbatch forec_all_maml.job
# FOREC models, need MAML
sbatch forec_all_forec.job
# Market aware models
sbatch forec_all_market_aware_base.job

```

## Global Model Experiments

```
# Market Unaware
python train_base_all.py  --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --batch_size 1024 --num_epoch 25 --exp_output forec_single_model/base-forec_single_model_concat.json
# MAML
python train_maml.py  --experiment_type single_model --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --num_epoch 25 --exp_output forec_single_model/maml-forec_single_model_concat.json
# FOREC
python train_forec.py  --experiment_type single_model --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat  --num_epoch 25 --exp_output forec_single_model/forec-forec_single_model_concat.json
# Market Aware
python train_base_all.py  --data_dir DATA2/proc_data --exp_name base_single_model --cuda --data_sampling_method concat --batch_size 1024 --num_epoch 25 --market_aware --exp_output forec_single_model/base-forec_single_model_concat_market_aware.json
```