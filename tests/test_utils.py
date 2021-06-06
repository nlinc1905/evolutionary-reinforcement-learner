import numpy as np

from utils import get_mean_and_standardized_rewards, mutate, update_params


def test_get_mean_and_standardized_rewards():

    # Assert that the mean reward for params 1,3,5 is 3.0
    test_array = np.array([1, 3, 5])
    mr, sr = get_mean_and_standardized_rewards(reward_per_offspring=test_array)
    assert mr == 3.0
    np.testing.assert_allclose(actual=sr, desired=[-1.22474487, 0., 1.22474487], rtol=1e-5)

    # Assert that, for a params array of all the same values, the mean equals one of the values,
    #   and the standardized rewards are an array of very bad values
    test_array = np.array([1, 1, 1])
    mr, sr = get_mean_and_standardized_rewards(reward_per_offspring=test_array)
    assert mr == 1
    np.testing.assert_allclose(actual=sr, desired=[-1e3, -1e3, -1e3], rtol=1e-5)


def test_mutate():
    # Assert that 2 sample arrays for reward and noise combine to an expected result
    test_reward_array = np.array([3, 5])
    test_noise_array = np.array([[1, 3, 5], [2, 4, 6]])
    r = mutate(standardized_reward=test_reward_array, noise_array=test_noise_array)
    np.testing.assert_array_equal(r, np.array([13, 29, 45]))


def test_update_params():
    # Assert that sample arrays for reward, noise, and params combine to an expected result
    test_reward_array = np.array([3, 5])
    test_noise_array = np.array([[1, 3, 5], [2, 4, 6]])
    r = update_params(
        params=np.array([5, 4, 3]),
        learning_rate=1e-3,
        sigma=0.1,
        noise_array=test_noise_array,
        standardized_reward_w_noise=mutate(
            standardized_reward=test_reward_array,
            noise_array=test_noise_array
        )
    )
    np.testing.assert_array_equal(r, np.array([5.065, 4.145, 3.225]))
