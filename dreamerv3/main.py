import importlib
import os
import pathlib
import sys
from functools import partial as bind

# wandb_key.txt 파일이 존재하면 읽어서 환경변수로 자동 설정
try:
  with open('wandb_key.txt', 'r') as f:
    os.environ['WANDB_API_KEY'] = f.read().strip()
except FileNotFoundError:
  pass

import wandb
_orig_wandb_video = wandb.Video
def _wandb_video_wrapper(data, *args, **kwargs):
  import numpy as np
  data = np.array(data)
  # If data is (T, C, H, W) and C=1, repeat to make it C=3
  if data.ndim == 4 and data.shape[1] == 1:
    data = np.repeat(data, 3, axis=1)
  if 'format' not in kwargs:
    kwargs['format'] = 'mp4'
  return _orig_wandb_video(data, *args, **kwargs)
wandb.Video = _wandb_video_wrapper

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml


def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))

_ANNOTATED_VIDEOS_CACHE = {}

def annotate_video_with_time(step, name, value):
  global _ANNOTATED_VIDEOS_CACHE
  cache_key = (step, name)
  if cache_key in _ANNOTATED_VIDEOS_CACHE:
    return _ANNOTATED_VIDEOS_CACHE[cache_key]
  
  import numpy as np
  from PIL import Image, ImageDraw, ImageFont

  if len(value.shape) != 4 or value.dtype != np.uint8:
    return value
  
  vid = value
  if vid.shape[-1] not in [1, 3, 4] and vid.shape[1] in [1, 3, 4]:
     vid = np.transpose(vid, [0, 2, 3, 1])
  
  T_frames, H, W, C = vid.shape
  if C == 1:
     vid = np.repeat(vid, 3, axis=-1)
     C = 3

  # Nearest neighbor upscale first to preserve Atari pixels exactly, while making text sharp
  scale = max(1, int(round(512 / H)))
  if scale > 1:
    vid = np.repeat(np.repeat(vid, scale, axis=1), scale, axis=2)
    H, W = H * scale, W * scale

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
  except:
    font = ImageFont.load_default()

  annotated = []
  for t in range(T_frames):
    frame = vid[t]
    if name == 'report/openloop/image':
      pad = 50
      new_frame = np.zeros((H + pad, W, C), dtype=np.uint8)
      new_frame[pad:, :, :] = frame
      img = Image.fromarray(new_frame)
      draw = ImageDraw.Draw(img)
      draw.text((W // 2 - 30, 10), f"T={t}", fill=(255, 255, 255), font=font)
      annotated.append(np.array(img))
    elif name == 'epstats/policy_image':
      pad = 140
      new_frame = np.zeros((H, W + pad, C), dtype=np.uint8)
      new_frame[:, :W, :] = frame
      img = Image.fromarray(new_frame)
      draw = ImageDraw.Draw(img)
      
      text = f"T={t}"
      import builtins
      rewards = getattr(builtins, 'last_epstats_rewards', None)
      if rewards is not None and t < len(rewards):
        cur_r = float(rewards[t])
        sum_r = float(np.sum(rewards[:t+1]))
        text += f"\n\nRew:\n{cur_r:+.2f}\n\nSum:\n{sum_r:+.2f}"
      
      draw.text((W + 10, H // 2 - 50), text, fill=(255, 255, 255), font=font)
      annotated.append(np.array(img))
      
  res = np.stack(annotated)
  _ANNOTATED_VIDEOS_CACHE[cache_key] = res
  
  # Clear old cache to avoid memory leak
  keys_to_remove = [k for k in _ANNOTATED_VIDEOS_CACHE if k[0] < step - 100]
  for k in keys_to_remove:
    del _ANNOTATED_VIDEOS_CACHE[k]
    
  return res

class LocalVideoMP4Output:
  def __init__(self, base_dir, run_name, fps=20):
    import pathlib
    self._dir = pathlib.Path(base_dir) / run_name
    self._dir.mkdir(parents=True, exist_ok=True)
    self._fps = fps

  def __call__(self, summaries):
    import imageio
    import numpy as np
    for step, name, value in summaries:
      if name in ('report/openloop/image', 'epstats/policy_image'):
        if len(value.shape) == 4:
          try:
            value = annotate_video_with_time(step, name, value)
          except Exception as e:
            print(f"Error annotating video {name}: {e}")
          vid = value
          if vid.shape[-1] not in [1, 3, 4] and vid.shape[1] in [1, 3, 4]:
             vid = np.transpose(vid, [0, 2, 3, 1])
          if vid.dtype != np.uint8:
            vid = (255 * np.clip(vid, 0, 1)).astype(np.uint8)
          
          scale = max(1, int(round(512 / vid.shape[1])))
          if scale > 1:
            vid = np.repeat(np.repeat(vid, scale, axis=1), scale, axis=2)
          
          safe_name = name.replace('/', '_')
          filename = self._dir / f"{step}_{safe_name}.mp4"
          try:
            imageio.mimsave(str(filename), vid, fps=self._fps, macro_block_size=1, quality=10, pixelformat='yuv444p')
          except Exception as e:
            print(f"Failed to save video: {e}")

class WandBOutputWrapper:
  def __init__(self, name, config=None):
    self._output = elements.logger.WandBOutput(name)
    if config is not None:
      import wandb
      if wandb.run is not None:
        try:
          wandb.config.update(dict(config), allow_val_change=True)
          import pathlib
          yaml_path = pathlib.Path(__file__).parent / 'configs.yaml'
          if yaml_path.exists():
            with open(yaml_path, 'r') as f:
              wandb.config.update({'configs.yaml': f.read()}, allow_val_change=True)
        except Exception as e:
          print(f'Failed to update wandb config: {e}')
  def __call__(self, summaries):
    new_summaries = []
    for step, name, value in summaries:
      if name in ('report/openloop/image', 'epstats/policy_image'):
        try:
          value = annotate_video_with_time(step, name, value)
        except Exception as e:
          print(f"Error annotating video {name}: {e}")
      
      if name.startswith('train/WorldModel/'):
        new_summaries.append((step, name.replace('train/WorldModel/', 'WorldModel/', 1), value))
      elif name.startswith('train/ActorCritic/'):
        new_summaries.append((step, name.replace('train/ActorCritic/', 'ActorCritic/', 1), value))
      else:
        new_summaries.append((step, name, value))
    self._output(tuple(new_summaries))

def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(WandBOutputWrapper(name, config))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
      
  # Add LocalVideoMP4Output
  run_name = logdir.split('/')[-1] if logdir else 'default_run'
  outputs.append(LocalVideoMP4Output('/home/jovyan/dowser-lora-datavol-1/choemj/dreamerv3/play_video', run_name, config.logger.fps))
  
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
