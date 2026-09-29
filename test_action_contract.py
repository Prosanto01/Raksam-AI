from agent.action_contract import validate_action


def test_rejects_unusable_click():
    action, error = validate_action({"action": "click"}, "pc")
    assert action is None and "coordinates or a visible target" in error


def test_normalizes_write_to_type():
    action, error = validate_action({"action": "write", "target": "hello"}, "pc")
    assert not error and action["action"] == "type"


def test_accepts_groundable_click():
    action, error = validate_action({"action": "click", "target": "Save"}, "pc")
    assert not error and action["target"] == "Save"


if __name__ == "__main__":
    test_rejects_unusable_click()
    test_normalizes_write_to_type()
    test_accepts_groundable_click()
    print("PASS: action contract")
