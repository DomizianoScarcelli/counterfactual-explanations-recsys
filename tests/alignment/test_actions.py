import pytest
from alignment.actions import Action, encode_action, encode_action_str, decode_action, print_action, is_legal

def test_encode_action_EncodesCorrectly_WhenValidInputs():
    assert encode_action(Action.SYNC, 42) == (Action.SYNC << 13) | 42
    assert encode_action(Action.ADD, 99) == (Action.ADD << 13) | 99
    assert encode_action(Action.DEL, 7) == (Action.DEL << 13) | 7

def test_encode_action_str_EncodesCorrectly_WhenValidStrings():
    assert encode_action_str("sync_42") == encode_action(Action.SYNC, 42)
    assert encode_action_str("add_15") == encode_action(Action.ADD, 15)
    assert encode_action_str("del_8") == encode_action(Action.DEL, 8)

def test_encode_action_str_RaisesAssertionError_WhenInvalidString():
    with pytest.raises(AssertionError, match="Action 'invalid_action' not supported"):
        encode_action_str("invalid_action")

def test_decode_action_DecodesCorrectly_WhenValidEncodedAction():
    assert decode_action(encode_action(Action.SYNC, 42)) == (Action.SYNC, 42)
    assert decode_action(encode_action(Action.ADD, 99)) == (Action.ADD, 99)
    assert decode_action(encode_action(Action.DEL, 7)) == (Action.DEL, 7)

def test_print_action_ReturnsCorrectString_WhenValidEncodedAction():
    assert print_action(encode_action(Action.SYNC, 42)) == "sync_42"
    assert print_action(encode_action(Action.ADD, 15)) == "add_15"
    assert print_action(encode_action(Action.DEL, 8)) == "del_8"

def test_is_legal_ReturnsTrue_WhenNoConflicts():
    prev_actions = {encode_action(Action.ADD, 3)}
    illegal_symbols = {5}
    assert is_legal(encode_action(Action.SYNC, 4), prev_actions, illegal_symbols) is True

def test_is_legal_ReturnsFalse_WhenIllegalSymbolUsed():
    prev_actions = {encode_action(Action.ADD, 3)}
    illegal_symbols = {4}
    assert is_legal(encode_action(Action.SYNC, 4), prev_actions, illegal_symbols) is False

def test_is_legal_ReturnsFalse_WhenConflictingActionExists():
    prev_actions = {encode_action(Action.ADD, 4)}
    illegal_symbols = set()
    assert is_legal(encode_action(Action.DEL, 4), prev_actions, illegal_symbols) is False

