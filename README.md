# CALog
Codebase paper: "CALog: Content Aware Log Anomaly Detection via Graph Neural Networks".

## Quick Start
- Use python vitural env.
- Python 3.11.8
- [PyTorch 2.2.2 https://pytorch.org/get-started/locally/)
- Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

- Install dependencies
```
pip install -r requirements.txt
```


## Dataset
Use public dataset on [Loghub](https://github.com/logpai/loghub?tab=readme-ov-file):


## Content aware for logs
```
python LFE.py \
    --gen_data ${DATA} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUT_DIR} \
    --strategy ${STRATEGY} \
    --n_shots ${NUM_SHOTS} \
    --n_grams ${N_GRAMS} \
    --neg_rate ${NEG_RATE} \
    --labeling_technique ${LABEL_METHOD} \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --num_train_epochs ${EPOCHS} \
    --do_train \
    --do_eval \
    --train_batch_size ${TRAIN_BATCH} \
    --eval_batch_size ${EVAL_BATCH} \
    --gradient_accumulation_steps ${GRAD_CUM_STEPS} \
    --ckpt_dir ${CKPT_DIR} \
    --seed ${SEED} \
    --overwrite_cache
```

## Generate graph
```
python graph_generation.py \
    --root ${ROOT} \
    --log_file ${DATA} \
    --inference_type ${INFERENCE} \
    --strategy ${STRATEGY} \
    --label_type node \
    --pretrained_model_name_or_path ${MODEL_PATH} \
    --interval ${INTERVAL} \
    --event_template 
```

## Train graph anomaly detection model
```
python main.py \
    --root ${ROOT} \
    --checkpoint_dir ${CKPT} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --model_type dynamic \
    --pretrained_model_path ${MODEL_PATH} \
    --lambda_seq ${LAMBDA} \
    --classification ${CLASSIFICATION} \
    --max_length ${MAX_LENGTH} \
    --lr ${LR} \
    --layers ${LAYERS} \
    --weight_decay ${WEIGHT_DECAY} \
    --do_train \
    --do_eval \
    --multi_granularity \
    --global_weight ${GLOBAL_WEIGHT} \
    --from_scratch
```


### References
Our code is developed based on [DiGCN](https://github.com/flyingtango/DiGCN).
