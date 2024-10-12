import pytest
from graph_search import Action, encode_action, decode_action, encode_action_str
from memory_profiler import memory_usage

@pytest.fixture
def action_map():
    return {
        (Action.ADD, 2588): "add_2588",
        (Action.DEL, 1244): "del_1244",
        (Action.SYNC, 1421): "sync_1421",
        (Action.ADD, 40): "add_40",
        (Action.DEL, 50): "del_50"
    }

@pytest.fixture
def actions():
    # Generating a long list of actions for testing, for example 10,000 actions
    return [f"{action_type}_{i}" for action_type in ["add", "del", "sync"] for i in range(10000)]

def test_action_encoding(action_map):
    for (action_type, number), expected_string in action_map.items():
        encoded = encode_action(action_type, number)
        decoded_action_type, decoded_number = decode_action(encoded)

        # Check if encoding and decoding are correct
        assert decoded_action_type == action_type
        assert decoded_number == number

        # Check if the expected string representation matches
        assert f"{expected_string}" == f"{expected_string}"

def test_action_memory_efficiency(actions):
    # Measure memory usage for string tuples
    def create_string_tuples():
        return tuple(f"{action_type}_{i}" for action_type in ["add", "del", "sync"] for i in range(10000))

    mem_usage_strings = memory_usage(create_string_tuples, interval=0.1) 
    string_memory_usage = (max(mem_usage_strings) - min(mem_usage_strings))  * 1024 * 1024

    def create_encoded_tuples():
        return tuple(encode_action_str(a) for a in actions)

    mem_usage_encoded = memory_usage(create_encoded_tuples, interval=0.1)
    encoded_memory_usage = (max(mem_usage_encoded) - min(mem_usage_encoded)) * 1024 * 1024

    # Assert that the memory usage of the string tuples is less than the encoded actions
    assert string_memory_usage > encoded_memory_usage, (
        f"String tuple memory usage {string_memory_usage} bytes"
        f"should be more than encoded actions memory usage {encoded_memory_usage} bytes"
    )
    print(f"String actions memory usage: {string_memory_usage}b")
    print(f"Encoded actions memory usage: {encoded_memory_usage}b")
