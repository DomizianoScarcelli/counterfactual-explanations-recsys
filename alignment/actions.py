from typing import Set
import os

if os.environ["LINE_PROFILE"] == "1":
    from line_profiler import profile


# Store actions as raw bits for memory efficiency
class Action:
    SYNC = 0b00  # 0
    DEL = 0b01  # 1
    ADD = 0b10  # 2


def encode_action(action_type: int, number: int) -> int:
    return (action_type << 13) | number  # 2 bits for action type, 13 bits for number


def encode_action_str(action: str) -> int:
    action_type = None
    number = None
    if "sync" in action:
        action_type = Action.SYNC
        number = int(action.replace("sync_", ""))
    elif "del" in action:
        action_type = Action.DEL
        number = int(action.replace("del_", ""))
    elif "add" in action:
        action_type = Action.ADD
        number = int(action.replace("add_", ""))

    assert action_type is not None, f"Action '{action}' not supported"
    assert number is not None, f"Number not extracted from action '{action}'"

    return encode_action(action_type, number)  # Use the encode_action function


def decode_action(encoded_action: int):
    action_type = (encoded_action >> 13) & 0b11  # Extract the action type (2 bits)
    number = encoded_action & 0x1FFF  # Extract the number (13 bits)
    return action_type, number


def act_str(action: int):
    if action == Action.SYNC:
        return "sync"
    if action == Action.ADD:
        return "add"
    if action == Action.DEL:
        return "del"


def print_action(encoded_action: int):
    action_type, e = decode_action(encoded_action)
    return f"{act_str(action_type)}_{e}"


def is_legal(action: int, prev_actions: Set[int], illegal_symbols: Set[int]) -> bool:
    _, e = decode_action(action)

    if e in illegal_symbols:
        return False
    add_e = encode_action(Action.ADD, e)
    del_e = encode_action(Action.DEL, e)
    sync_e = encode_action(Action.SYNC, e)
    actions = {add_e, del_e, sync_e}
    other_actions = actions - {action}
    return not (prev_actions & other_actions)
