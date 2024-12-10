import pytest

from utils_classes.Cached import Cached


@pytest.fixture(autouse=True)
def reset_cached_instances():
    """
    Fixture to reset the Cached class's _instances dictionary before each test.
    """
    Cached._instances = {}

# Sample load function for testing
def mock_load_function(path: str):
    return f"Mock data loaded from {path}"

def test_Cached_ReturnsSameInstance_WhenPathIsTheSame():
    """
    Test that the Cached class returns the same instance for the same path.
    """
    path = "path/to/file1.txt"
    cache1 = Cached(path, mock_load_function)
    cache2 = Cached(path, mock_load_function)

    assert cache1 is cache2
    assert cache1.get_data() == "Mock data loaded from path/to/file1.txt"

def test_Cached_CreatesDifferentInstances_WhenPathIsDifferent():
    """
    Test that the Cached class creates different instances for different paths.
    """
    path1 = "path/to/file1.txt"
    path2 = "path/to/file2.txt"

    cache1 = Cached(path1, mock_load_function)
    cache2 = Cached(path2, mock_load_function)

    assert cache1 is not cache2
    assert cache1.get_data() == "Mock data loaded from path/to/file1.txt"
    assert cache2.get_data() == "Mock data loaded from path/to/file2.txt"

def test_Cached_CallsLoadFunctionOnce_WhenPathIsTheSame():
    """
    Test that the load function is called only once for the same path.
    """
    calls = []

    def tracking_load_function(path: str):
        calls.append(path)
        return f"Data loaded from {path}"

    path = "path/to/file1.txt"
    cache1 = Cached(path, tracking_load_function)
    cache2 = Cached(path, tracking_load_function)

    assert len(calls) == 1
    assert cache1.get_data() == "Data loaded from path/to/file1.txt"
    assert cache2.get_data() == "Data loaded from path/to/file1.txt"

def test_Cached_RaisesError_WhenLoadFunctionIsInvalid():
    """
    Test that providing an invalid (non-callable) load function raises a ValueError.
    """
    path = "path/to/file1.txt"
    invalid_load_fn = "not_a_function"

    with pytest.raises(ValueError, match="load_fn must be a callable function."):
        Cached(path, invalid_load_fn)

def test_Cached_DataIntegrity_WhenInstanceIsAccessed():
    """
    Test that data loaded by a Cached instance is correct and remains intact.
    """
    path = "path/to/file1.txt"
    expected_data = "Mock data loaded from path/to/file1.txt"

    cache = Cached(path, mock_load_function)
    assert cache.get_data() == expected_data

def test_Cached_PreventsReinitialization_WhenInstanceExists():
    """
    Test that the `_initialized` flag prevents reinitialization of an existing instance.
    """
    path = "path/to/file1.txt"

    cache = Cached(path, mock_load_function)
    assert hasattr(cache, "_initialized")
    assert cache._initialized is True

    # Attempt reinitialization with a different load function
    new_load_fn = lambda p: "New data"
    cache_reused = Cached(path, new_load_fn)
    assert cache_reused.get_data() == "Mock data loaded from path/to/file1.txt"

def test_Cached_HandlesMultiplePaths_WhenInstancesAreCreated():
    """
    Test that creating multiple Cached instances with different paths works correctly.
    """
    path1 = "path/to/file1.txt"
    path2 = "path/to/file2.txt"

    cache1 = Cached(path1, mock_load_function)
    cache2 = Cached(path2, mock_load_function)

    assert cache1.get_data() == "Mock data loaded from path/to/file1.txt"
    assert cache2.get_data() == "Mock data loaded from path/to/file2.txt"

def test_Cached_RaisesException_WhenPathIsEmpty():
    """
    Test that providing an empty path raises an appropriate exception.
    """
    with pytest.raises(Exception):  # Adjust exception type based on implementation
        Cached("", mock_load_function)
