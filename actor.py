import os
import pickle
import subprocess
import time
from argparse import ArgumentParser
from collections import deque
from itertools import count
from multiprocessing import Process, Array
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import zmq
from pyarrow import serialize

from common import init_components, load_yaml_config, save_yaml_config, create_experiment_dir
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='The game environment')
parser.add_argument('--num_steps', type=int, default=10000000, help='The number of total training steps')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--num_replicas', type=int, default=1, help='The number of actors')
parser.add_argument('--model', type=str, default='accnn', help='Training model')
parser.add_argument('--max_steps_per_update', type=int, default=128,
                    help='The maximum number of steps between each update')
parser.add_argument('--exp_path', type=str, default=None,
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=10, help='Number of recent checkpoint files to be saved')
parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum length of trajectory')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU to sample every action')
parser.add_argument('--num_envs', type=int, default=10, help='The number of environment copies')


def run_one_agent(index, args, unknown_args, actor_status):
    from tensorflow.keras.backend import set_session
    import tensorflow.compat.v1 as tf

    # Set 'allow_growth'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Connect to learner
    context = zmq.Context()
    context.linger = 0  # For removing linger behavior
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    # Initialize environment and agent instance
    env, agent = init_components(args, unknown_args)

    # Configure logging only in one process
    if index == 0:
        logger.configure(str(args.log_path))
        save_yaml_config(args.exp_path / 'config.yaml', args, 'actor', agent)
    else:
        logger.configure(str(args.log_path), format_strs=[])

    # Initialize values
    model_id = -1
    episode_infos = deque(maxlen=100)
    num_episode = 0
    state = env.reset()

    n_batch = args.num_envs * args.max_steps_per_update

    nupdates = args.num_steps // (n_batch * args.num_replicas)

    for update in range(1, nupdates + 1):
        # Do some updates
        agent.update_sampling(update, nupdates)

        mb_states, mb_actions, mb_rewards, mb_dones, mb_extras = [], [], [], [], []
        start_time = time.time()
        for _ in range(args.max_steps_per_update):

            mb_states.append(state)

            # Sample action
            action, extra_data = agent.sample(state)
            state, reward, done, info = env.step(action)

            mb_actions.append(action)
            mb_rewards.append(reward)
            mb_dones.append(done)
            mb_extras.append(extra_data)

            for info_i in info:
                maybeepinfo = info_i.get('episode')
                if maybeepinfo:
                    episode_infos.append(maybeepinfo)
                    num_episode += 1

        mb_states = np.asarray(mb_states, dtype=state.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        data = agent.prepare_training_data([mb_states, mb_actions, mb_rewards, mb_dones, state, mb_extras])
        post_processed_data = agent.post_process_training_data(data)
        socket.send(serialize(post_processed_data).to_buffer())
        socket.recv()

        send_data_interval = time.time() - start_time
        # Log information
        logger.record_tabular("steps", update * n_batch)
        logger.record_tabular("episodes", num_episode)
        logger.record_tabular("mean 100 episode reward",
                              round(np.mean([epinfo['reward'] for epinfo in episode_infos]), 2))
        logger.record_tabular("mean 100 episode length",
                              round(np.mean([epinfo['length'] for epinfo in episode_infos]), 2))
        logger.record_tabular("send data interval", send_data_interval)
        logger.record_tabular("send data fps", n_batch // send_data_interval)
        logger.record_tabular("total steps", nupdates * n_batch)
        logger.dump_tabular()

        # Update weights
        new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            agent.set_weights(new_weights)

    actor_status[index] = 1


def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
                    f.write(weights)

                if model_id > args.num_saved_ckpt:
                    os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.{args.alg}.{args.env}.ckpt')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # All actors finished works
                return

            # For not cpu-intensive
            time.sleep(1)


def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = pickle.load(f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'actor')

    # Create experiment directory
    create_experiment_dir(args, 'ACTOR-')

    args.ckpt_path = args.exp_path / 'ckpt'
    args.log_path = args.exp_path / 'log'
    args.ckpt_path.mkdir()
    args.log_path.mkdir()

    # Record commit hash
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Disable GPU
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Running status of actors
    actor_status = Array('i', [0] * args.num_replicas)

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(args, actor_status))
    subscriber.start()

    def exit_wrapper(index, *x, **kw):
        """Exit all agents on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_agent(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(agents):
                    if _i != index:
                        _p.terminate()
                    actor_status[_i] = 1

    agents = []
    for i in range(args.num_replicas):
        p = Process(target=exit_wrapper, args=(i, args, unknown_args, actor_status))
        p.start()
        os.system(f'taskset -p -c {i % os.cpu_count()} {p.pid}')  # For CPU affinity

        agents.append(p)

    for agent in agents:
        agent.join()

    subscriber.join()


if __name__ == '__main__':
    main()
