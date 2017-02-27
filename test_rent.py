from homework2_rent import score_rent


def test_rent():
	expected_outcome = 0.48
	assert score_rent() > expected_outcome

test_rent()