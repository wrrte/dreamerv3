import os
import pathlib
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp

_CACHE = {}

def get_latent_state(agent, obs):
    carry = agent.init_policy(batch_size=1)
    enc_carry, dyn_carry, dec_carry, prevact = carry
    _, _, tokens = agent.enc(enc_carry, obs, reset=obs['is_first'], training=False, single=True)
    _, _, feat = agent.dyn.observe(dyn_carry, tokens, prevact, reset=obs['is_first'], training=False, single=True)
    latent_tensor = agent.feat2tensor(feat)
    return latent_tensor.squeeze()

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert('RGB').resize((64, 64))
    img_arr = np.array(img)[None, ...] # (1, 64, 64, 3)
    obs = {
        'image': img_arr,
        'is_first': np.array([True]),
        'is_terminal': np.array([False])
    }
    return obs

def run_probing(agent):
    metrics = {}
    
    base_dir = pathlib.Path(__file__).parent.parent / 'Probing_Images' / 'seaquest'
    selected_divers_dir = base_dir / '64_selected_divers'
    sea_divers_dir = base_dir / '64_sea_divers'
    
    if not selected_divers_dir.exists() or not sea_divers_dir.exists():
        return metrics

    # 1. 64_selected_divers
    if 'selected_divers' not in _CACHE:
        groups = {}
        for f in os.listdir(selected_divers_dir):
            if f.endswith('.png'):
                parts = f.split('_')
                frame_prefix = f"{parts[0]}_{parts[1]}"
                diver_count = int(parts[3].split('.')[0])
                if frame_prefix not in groups:
                    groups[frame_prefix] = {}
                groups[frame_prefix][diver_count] = selected_divers_dir / f
        
        loaded_groups = {}
        for prefix, files in groups.items():
            loaded_groups[prefix] = [load_and_preprocess(files[i]) for i in range(7) if i in files]
        _CACHE['selected_divers'] = loaded_groups

    # 2. 64_sea_divers (DiverMoves only)
    if 'sea_divers' not in _CACHE:
        groups = {}
        distances = [10, 25, 40, 55, 70]
        for f in os.listdir(sea_divers_dir):
            if 'DiverMoves' in f and f.endswith('.png'):
                parts = f.split('_')
                type_prefix = parts[0]
                dist = int(parts[2].replace('dist', '').replace('.png', ''))
                if type_prefix not in groups:
                    groups[type_prefix] = {}
                groups[type_prefix][dist] = sea_divers_dir / f
        
        loaded_groups = {}
        for prefix, files in groups.items():
            loaded_groups[prefix] = [load_and_preprocess(files[d]) for d in distances if d in files]
        _CACHE['sea_divers'] = loaded_groups

    # Evaluate 64_selected_divers
    selected_groups = _CACHE['selected_divers']
    all_selected_matrices = []
    num_divers = 7
    
    for prefix, obs_list in selected_groups.items():
        if len(obs_list) != num_divers: continue
        latents = [get_latent_state(agent, obs) for obs in obs_list] # list of (D,)
        latents_stack = jnp.stack(latents) # (7, D)
        norms = jnp.linalg.norm(latents_stack, axis=1, keepdims=True)
        normalized = latents_stack / (norms + 1e-8)
        sim_matrix = jnp.dot(normalized, normalized.T) # (7, 7)
        all_selected_matrices.append(sim_matrix)
        
    if all_selected_matrices:
        avg_matrix = jnp.mean(jnp.stack(all_selected_matrices), axis=0) # (7, 7)
        heatmap_img = jnp.clip(avg_matrix, 0.0, 1.0)
        heatmap_img = (heatmap_img * 255).astype(jnp.uint8)
        heatmap_img = heatmap_img[..., None]
        metrics['probe_selected/heatmap'] = heatmap_img 
        
        for i in range(num_divers):
            for j in range(i + 1, num_divers):
                metrics[f'probe_selected/pair_{i}_vs_{j}'] = avg_matrix[i, j]
                
        for delta in range(1, num_divers):
            diag = jnp.diag(avg_matrix, k=delta)
            metrics[f'probe_selected/delta_{delta}'] = jnp.mean(diag)

    # Evaluate 64_sea_divers
    sea_groups = _CACHE['sea_divers']
    all_sea_matrices = []
    num_dists = 5
    
    for prefix, obs_list in sea_groups.items():
        if len(obs_list) != num_dists: continue
        latents = [get_latent_state(agent, obs) for obs in obs_list]
        latents_stack = jnp.stack(latents)
        norms = jnp.linalg.norm(latents_stack, axis=1, keepdims=True)
        normalized = latents_stack / (norms + 1e-8)
        sim_matrix = jnp.dot(normalized, normalized.T)
        all_sea_matrices.append(sim_matrix)
        
    if all_sea_matrices:
        avg_matrix = jnp.mean(jnp.stack(all_sea_matrices), axis=0)
        heatmap_img = jnp.clip(avg_matrix, 0.0, 1.0)
        heatmap_img = (heatmap_img * 255).astype(jnp.uint8)
        heatmap_img = heatmap_img[..., None]
        metrics['probe_sea/heatmap'] = heatmap_img
        
        for i in range(num_dists):
            for j in range(i + 1, num_dists):
                metrics[f'probe_sea/pair_{i}_vs_{j}'] = avg_matrix[i, j]
                
        for delta in range(1, num_dists):
            diag = jnp.diag(avg_matrix, k=delta)
            metrics[f'probe_sea/delta_{delta}'] = jnp.mean(diag)

    return metrics
