from bandit_code.simple_epsilon_gready import EpsilonGreedy

def check_value_in_range(value):
    if 0 <= value < 10:
        return True
    else:
        return False


def test_epsilon_gready_picked_arm():

    epsilon_gready = EpsilonGreedy()

    epsilon_gready.pre_populate_counts_and_values(n_arms=10)
    picked_arm = epsilon_gready.pick_arm()

    assert check_value_in_range(picked_arm) == True
