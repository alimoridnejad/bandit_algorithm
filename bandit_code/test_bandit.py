from bandit_code.draft import EpsilonGreedy, Learner


def test_eplison_greedy():

    epsilon_greedy = EpsilonGreedy(Learner)

    computed_output = epsilon_greedy.run_one_step()
    expected_output = 3

    assert expected_output == computed_output