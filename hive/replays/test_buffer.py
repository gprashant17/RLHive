import pytest
import os

import numpy as np
from hive import envs, replays

environment = envs.GymEnv('CartPole-v1')
rng = np.random.default_rng(seed=100)
buffer = replays.CircularReplayBuffer(rng, size=500, compress=True)

observation, _ = environment.reset()
for i in range(400):
    action = rng.integers(environment._env_spec.act_dim)
    next_observation, reward, done, turn, info = environment.step(action)
    buffer.add((observation,
                action,
                reward,
                next_observation,
                done))
    observation = next_observation
    if done:
        observation, _ = environment.reset()


def test_add_to_buffer():
    """
        test adding one transition to the buffer
    """

    observation, _ = environment.reset()
    action = rng.integers(environment._env_spec.act_dim)
    next_observation, reward, done, turn, info = environment.step(action)
    buffer.add((observation,
                action,
                reward,
                next_observation,
                done))
    assert buffer.size() == 401


@pytest.mark.parametrize("batch_size", [32])
def test_sample_from_buffer(batch_size):
    """
    test sampling a batch from the buffer
    """

    batch = buffer.sample(batch_size=batch_size)
    assert batch["observations"].shape == (batch_size, 4)
    assert batch["actions"].shape == (batch_size,)
    assert batch["rewards"].shape == (batch_size,)
    assert batch["next_observations"].shape == (batch_size, 4)
    assert batch["done"].shape == (batch_size,)


def test_saving_buffer(tmpdir):
    """
        test sampling a batch from the buffer
    """

    buffer.save(tmpdir.mkdir("saved_test_buffer"))
    assert os.path.exists(tmpdir / "saved_test_buffer") is True


@pytest.mark.parametrize("batch_size", [32])
def test_loading_buffer(tmpdir, batch_size):
    """
        test sampling a batch from the buffer
    """

    buffer.save(tmpdir / "saved_test_buffer")
    assert os.path.exists(tmpdir / "saved_test_buffer") is True

    buffer_loaded = replays.CircularReplayBuffer(rng, size=500, compress=True)
    buffer_loaded.load(tmpdir / "saved_test_buffer")

    batch = buffer_loaded.sample(batch_size)
    assert batch["observations"].shape == (batch_size, 4)
    assert batch["actions"].shape == (batch_size,)
    assert batch["rewards"].shape == (batch_size,)
    assert batch["next_observations"].shape == (batch_size, 4)
    assert batch["done"].shape == (batch_size,)

