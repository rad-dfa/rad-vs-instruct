import os
import jax
import yaml
import argparse
import itertools
import jax.numpy as jnp
from functools import partial
from rad_embeddings import Encoder, EncoderModule
from flax.traverse_util import flatten_dict
from dfa_gym import TokenEnv, DFAWrapper
import flax.serialization as serialization
from dfax.samplers import ReachSampler, ReachAvoidSampler, RADSampler
from train import ActorCritic
from ppo import batchify
from collections import Counter
from dfax import data2sampler


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Train TokenEnv policy")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="List of seeds for testing (must match trained checkpoints)"
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="dataset.pkl",
        help="Dataset used for training (default: dataset.pkl)"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="dataset.pkl",
        help="Dataset to be used for testing (default: dataset.pkl)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="storage",
        help="Directory for the params (default: storage)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of samples (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        help="Batch size (default: n)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="rad",
        help="Type of embeddings to use rad, instruct, or combine (default: rad)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results on a comma seperated line"
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    if batch_size == -1:
        batch_size = args.n

    sampler = data2sampler(args.test_dataset)
    train_dataset = os.path.splitext(os.path.basename(args.train_dataset))[0]
    test_dataset = os.path.splitext(os.path.basename(args.test_dataset))[0]

    n_seeds = len(args.seeds)
    success_rate_list = jnp.zeros((n_seeds,))
    avg_len_list = jnp.zeros((n_seeds,))
    avg_reward_list = jnp.zeros((n_seeds,))
    avg_disc_return_list = jnp.zeros((n_seeds,))

    for i, seed in enumerate(args.seeds):

        key = jax.random.PRNGKey(seed + 100)

        token_env = TokenEnv(
            n_agents=1,
            max_steps_in_episode=100,
            fixed_map_seed=seed
        )

        if args.type == "instruct" or args.type == "combine":
            env = DFAWrapper(
                env=token_env,
                gamma=None,
                sampler=sampler,
                binary_reward=False,
                embedder=sampler.embed,
                embedding_dim=sampler.embd_dim,
                combine_embed=args.type == "combine"
            )

            rad_str = args.type
            encoder = Encoder(
                max_size=env.sampler.max_size,
                n_tokens=token_env.n_tokens,
                seed=seed
            ) if args.type == "combine" else None

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
                seed=seed
            )

        network = ActorCritic(
            action_dim=env.action_space(env.agents[0]).n,
            encoder=encoder,
            n_agents=env.num_agents,
            combine=args.type == "combine"
        )

        key, subkey = jax.random.split(key)
        init_x = env.observation_space(env.agents[0]).sample(subkey)
        key, subkey = jax.random.split(key)
        params = network.init(subkey, init_x)

        with open(f"{args.model_dir}/policy_params_seed_{seed}_dataset_{train_dataset}_{rad_str}.msgpack", "rb") as f:
            params = serialization.from_bytes(params, f.read())

        @partial(jax.jit, static_argnums=(0, 1))
        def run_episode(env, network, params, key):
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey)

            carry = {
                "key": key,
                "obs": obs,
                "state": state,
                "done": False,
                "success": 0,
                "ep_len": 0,
                "ep_reward": 0.0,
                "ep_discounted_return": 0.0,
                "discount": 1.0,
            }

            def cond_fn(carry):
                return ~carry["done"]

            def body_fn(carry):
                key, subkey = jax.random.split(carry["key"])
                pi, value = network.apply(params, batchify(carry["obs"], env.agents))
                actions = pi.sample(seed=subkey)
                actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
                key, subkey = jax.random.split(key)
                obs, state, rewards, dones, infos = env.step(subkey, carry["state"], actions)

                done = dones["__all__"]
                rewards_arr = jnp.array([rewards[a] for a in env.agents])
                success = carry["success"] + jnp.all(rewards_arr > 0) * done
                ep_reward = carry["ep_reward"] + jnp.mean(rewards_arr)
                ep_discounted_return = carry["ep_discounted_return"] + carry["discount"] * jnp.mean(rewards_arr)
                discount = carry["discount"] * (env.gamma if env.gamma is not None else 1.0)
                ep_len = carry["ep_len"] + 1

                return {
                    "key": key,
                    "obs": obs,
                    "state": state,
                    "done": done,
                    "success": success,
                    "ep_len": ep_len,
                    "ep_reward": ep_reward,
                    "ep_discounted_return": ep_discounted_return,
                    "discount": discount,
                }

            final_carry = jax.lax.while_loop(cond_fn, body_fn, carry)
            return (
                final_carry["success"],
                final_carry["ep_len"],
                final_carry["ep_reward"],
                final_carry["ep_discounted_return"]
            )

        @partial(jax.jit, static_argnums=(0, 1))
        def run_episodes(env, network, params, keys):
            return jax.vmap(run_episode, (None, None, None, 0))(env, network, params, keys)

        def batched_vmap_run(env, network, params, keys, batch_size):
            n = keys.shape[0]
            results = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_keys = keys[start:end]
                batch_results = run_episodes(env, network, params, batch_keys)
                results.append(batch_results)
            results = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *results)
            return results

        keys = jax.random.split(key, args.n)
        results = batched_vmap_run(env, network, params, keys, batch_size=batch_size)

        success_counts, ep_lens, ep_rewards, ep_disc_returns = results

        success_rate = jnp.mean(success_counts)
        avg_len = jnp.mean(ep_lens)
        avg_reward = jnp.mean(ep_rewards)
        avg_disc_return = jnp.mean(ep_disc_returns)

        success_rate_list = success_rate_list.at[i].set(success_rate)
        avg_len_list = avg_len_list.at[i].set(avg_len)
        avg_reward_list = avg_reward_list.at[i].set(avg_reward)
        avg_disc_return_list = avg_disc_return_list.at[i].set(avg_disc_return)

    success_rate_mean = jnp.mean(success_rate_list)
    success_rate_std = jnp.std(success_rate_list)

    avg_len_mean = jnp.mean(avg_len_list)
    avg_len_std = jnp.std(avg_len_list)

    avg_reward_mean = jnp.mean(avg_reward_list)
    avg_reward_std = jnp.std(avg_reward_list)

    avg_disc_return_mean = jnp.mean(avg_disc_return_list)
    avg_disc_return_std = jnp.std(avg_disc_return_list)

    if args.csv:
        print(f"{rad_str}, {train_dataset}, {test_dataset}, {success_rate_mean} +/- {success_rate_std}, {avg_len_mean} +/- {avg_len_std}, {avg_reward_mean} +/- {avg_reward_std}, {avg_disc_return_mean} +/- {avg_disc_return_std}")
    else:
        print(f"Test completed for {n_seeds} seeds.")
        print(f"Success rate: {success_rate_mean:.2f} +/- {success_rate_std:.2f}")
        print(f"Average episode length: {avg_len_mean:.2f} +/- {avg_len_std:.2f}")
        print(f"Average episode reward: {avg_reward_mean:.2f} +/- {avg_reward_std:.2f}")
        print(f"Average episode discounted return: {avg_disc_return_mean:.2f} +/- {avg_disc_return_std:.2f}")

