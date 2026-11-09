import fuzzshrink


def test_package_imports_and_has_version():
    assert fuzzshrink.__version__.count(".") == 2
