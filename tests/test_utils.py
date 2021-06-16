import numpy as np

from utils import (
    get_mean_and_standardized_rewards,
    mutate,
    compute_centered_ranks,
    update_params,
    hamming_dist,
    calculate_nas_score,
)


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


def test_compute_centered_ranks():
    # Assert that a given input produces its expected output
    test_array = np.array([-1, 0, 3, 15])
    y = compute_centered_ranks(x=test_array)
    expected_output = np.array([-0.5, -0.166667,  0.166667,  0.5])
    np.testing.assert_allclose(actual=y, desired=expected_output, atol=1e-5)


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


def test_hamming_dist():
    # Assert that distance matrix equals what is expected for a given array
    test_array = np.array([[0, 3, 5], [2, 0, 6]])
    hamming = hamming_dist(arr=test_array)
    expected = np.array([
        [2, 0, 1],
        [0, 2, 1],
        [1, 1, 2]
    ])
    np.testing.assert_equal(actual=hamming, desired=expected)


def test_calculate_nas_score():
    # Assert that the score equals what is expected for a given array
    test_array = np.array([[0, 3, 5], [2, 0, 6]])
    hamming = hamming_dist(arr=test_array)
    score = calculate_nas_score(arr=test_array, hamming_dist_matrix=hamming)
    assert round(score, 4) == 1.0986
