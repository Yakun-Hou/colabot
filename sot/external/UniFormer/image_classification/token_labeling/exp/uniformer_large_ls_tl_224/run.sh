work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python3 -m torch.distributed.launch --nproc_per_node=8 main.py your_path_to_data \
  --model uniformer_large_ls \
  --batch-size 128 \
  --apex-amp \
  --lr 1.2e-3 \
  --img-size 224 \
  --drop-path 0.4 \
  --token-label \
  --token-label-data your_path_to_label_data \
  --token-label-size 7 \
  --model-ema \
  --output ${work_path} \
  2>&1 | tee ${work_path}/log.txt
