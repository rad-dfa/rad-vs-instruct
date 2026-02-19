import os
import jax
import wandb
import distrax
import argparse
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from ppo import make_train
from wrappers import LogWrapper
from dfax import list2batch, batch2graph, data2sampler
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from flax.traverse_util import flatten_dict
from rad_embeddings import Encoder, EncoderModule
from flax.linen.initializers import constant, orthogonal
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler, DataSampler

class ActorCritic(nn.Module):
    action_dim: int
    encoder: Encoder | None
    n_agents: int
    deterministic: bool = False
    padding: str = "VALID"
    combine: bool = False

    @nn.compact
    def __call__(self, batch):

        obs_batch = batch["obs"]
        if obs_batch.ndim == 3: # (C, H, W)
            obs_batch = obs_batch[None, ...] # -> (1, C, H, W)
        elif obs_batch.ndim != 4:
            raise ValueError(f"Expected (C, H, W) or (B, C, H, W), got {obs_batch.shape} for obs")
        obs_batch = jnp.transpose(obs_batch, (0, 2, 3, 1)) # -> (B, H, W, C)

        obs_feat = nn.Sequential([
            nn.Conv(16, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(32, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            nn.Conv(64, (2, 2), padding=self.padding, kernel_init=orthogonal(np.sqrt(2))),
            nn.relu,
            lambda x: x.reshape((x.shape[0], -1)),
        ])(obs_batch)

        if self.encoder is not None:
            dfa = batch2graph(batch["dfa"])
            rad = self.encoder(dfa)
            dfa_feat = rad
            if self.combine:
                emb = batch["emb"].reshape((batch["emb"].shape[0], -1))
                dfa_feat = jnp.concatenate([rad, emb], axis=-1)
        else:
            emb = batch["emb"].reshape((batch["emb"].shape[0], -1))
            dfa_feat = emb

        feat = jnp.concatenate([obs_feat, dfa_feat], axis=-1)

        value = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])(feat)

        logits = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        ])(feat)

        if self.deterministic:
            action = jnp.argmax(logits, axis=-1)
            return action, jnp.squeeze(value, axis=-1)
        else:
            pi = distrax.Categorical(logits=logits)
            return pi, jnp.squeeze(value, axis=-1)


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ANNEAL_LR": False,
    }

    parser = argparse.ArgumentParser(description="Train DFA-conditioned TokenEnv policy")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for PRNGKey (default: 42)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset.pkl",
        help="Dataset for sampling tasks (default: dataset.pkl)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory for saving the trained encoder (default: storage)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print logs"
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Use instruct embeddings"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine embeddings"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log.csv",
        help="Log csv name (default: log.csv)"
    )
    args = parser.parse_args()

    config["DEBUG"] = args.debug
    config["WANDB"] = args.wandb
    config["LOG"] = args.log

    if config["WANDB"]: # export WANDB_API_KEY="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"; uv run wandb login
        wandb.init(
            entity="beyazit-y-berkeley-eecs",
            project="rad-vs-instruct",
            config=config
        )

    key = jax.random.PRNGKey(args.seed)

    sampler = data2sampler(args.dataset)

    token_env = TokenEnv(
        n_agents=1,
        max_steps_in_episode=100,
        fixed_map_seed=args.seed
    )

    if args.instruct or args.combine:
        env = DFAWrapper(
            env=token_env,
            gamma=None,
            sampler=sampler,
            binary_reward=False,
            embedder=sampler.embed,
            embedding_dim=sampler.embd_dim,
            combine_embed=args.combine
        )

        rad_str = "combine" if args.combine else "instruct"
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=token_env.n_tokens,
            seed=args.seed
        ) if args.combine else None

    else:
        env = DFAWrapper(
            env=token_env,
            gamma=None,
            sampler=sampler,
            binary_reward=False,
        )
        rad_str = "rad"
        encoder = Encoder(
            max_size=env.sampler.max_size,
            n_tokens=token_env.n_tokens,
            seed=args.seed
        )

    env = LogWrapper(env=env, config=config)

    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).n,
        encoder=encoder,
        n_agents=env.num_agents,
        combine=args.combine
    )

    if config["DEBUG"]:
        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)
        flat = flatten_dict(params, sep="/")
        total = 0
        for k, v in flat.items():
            count = v.size
            total += count
            print(f"{k:60} {v.shape} {v.dtype} ({count:,} params)")
        print(f"\nTotal parameters: {total:,}")
    
    train_jit = jax.jit(make_train(config, env, network))
    out = train_jit(key)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = os.path.splitext(os.path.basename(args.dataset))[0]
    trained_params = out["runner_state"][0].params
    with open(f"{args.save_dir}/policy_params_seed_{args.seed}_dataset_{dataset}_{rad_str}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(trained_params))

    if config["WANDB"]:
        wandb.finish()

