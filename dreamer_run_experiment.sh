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

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/alien_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_alien \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/amidar_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_amidar \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/assault_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_assault \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/asterix_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_asterix \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/bankheist_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_bankheist \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/battlezone_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_battlezone \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/boxing_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_boxing \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/breakout_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_breakout \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/crazyclimber_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_crazyclimber \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/demonattack_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_demonattack \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/freeway_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_freeway \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/gopher_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_gopher \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/jamesbond_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_jamesbond \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/kangaroo_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_kangaroo \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/krull_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_krull \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/kungfumaster_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_kungfumaster \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/mspacman_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_mspacman \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/pong_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_pong \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/privateeye_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_privateeye \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/qbert_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_qbert \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/roadrunner_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_roadrunner \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000

python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/upndown_$(date +%Y%m%d_%H%M%S) \
  --configs atari \
  --task atari_upndown \
  --logger.outputs jsonl,scope,wandb \
  --run.train_ratio 32 \
  --run.steps 100000
