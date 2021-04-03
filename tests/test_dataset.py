from dumbbells.dataset import Dataset


def test_init():
    """Tests the initialization of a Dataset"""
    assert len(dataset.memory) == 0
    assert dataset.memory_size == memory_size
    assert dataset.position == 0
    assert len(dataset.state_space) == 2
    assert dataset.reward_space == [-1, 0]


def test_step():
    """Tests the step(), get_state(), and pushMem() methods"""

    prev_state = dataset.get_state()
    result = dataset.step(2)
    assert result[1] == -1
    assert prev_state[0] != dataset.get_state()[0]
    assert prev_state[1] != dataset.get_state()[1]
    assert len(dataset.memory) == 1
    assert dataset.position == 1


def test_memory():
    """Tests the __len__() and __getitem__() methods"""

    prev_state = dataset.get_state()
    assert dataset.__len__() == 1
    result = dataset.step(0)
    # (prev_state, 0, result[1], result[0])
    image = dataset.__getitem__(1)
    assert image[0][0] == prev_state[0]
    assert image[0][1] == prev_state[1]
    assert image[1] == 0
    assert image[2] == result[1]
    assert image[3][0] == result[0][0]
    assert image[3][1] == result[0][1]
    assert dataset.__len__() == 2


if __name__ == "__main__":

    print("\nStarting dataset tests using 'MountainCar-v0' environment...\n")

    # Define some global variables for use in the tests
    game = "MountainCar-v0"
    memory_size = 5
    dataset = Dataset(game, memory_size)

    # Test suite
    test_init()
    test_step()
    test_memory()

    # Tests completed
    print(
        "\n************************\nCompleted Dataset.py Tests\n************************\n"
    )
