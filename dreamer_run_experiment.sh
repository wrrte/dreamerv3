export CUDA_VISIBLE_DEVICES=1

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/seaquest_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_seaquest \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/hero_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_hero \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/frostbite_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_frostbite \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/choppercommand_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_choppercommand \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000
