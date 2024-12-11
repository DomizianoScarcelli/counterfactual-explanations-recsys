from typing import Callable

#TODO: put generic type
class Cached:
    """
    A class to implement a caching mechanism for loading and storing data based on unique paths.
    Ensures that for each unique path, only one instance of the class exists (singleton behavior per path).

    Attributes:
        _instances (dict): A class-level dictionary to store instances of the class, keyed by file path.

    Methods:
        __new__(cls, path):
            Controls the creation of new instances. If an instance for the given path already exists, 
            it returns the cached instance; otherwise, it creates and caches a new instance.

        __init__(self, path):
            Initializes the instance by loading the data from the specified path.
            Ensures initialization is done only once per instance.

        _load_data(self, path):
            Loads data from the specified path. This method should be overridden with actual
            data loading logic (e.g., reading a file).

        get_data(self):
            Retrieves the loaded data for the given instance.
    """

    _instances = {}  # Class-level dictionary to store instances by path

    def __new__(cls, path: str, load_fn: Callable):
        """
        Ensures a singleton-like behavior for instances based on unique file paths.

        Args:
            path (str): The file path for which the instance is created.

        Returns:
            Cached: An existing or newly created instance of the Cached class.
        """
        if path in cls._instances:
            return cls._instances[path]
        instance = super().__new__(cls)
        cls._instances[path] = instance
        return instance

    def __init__(self, path: str, load_fn: Callable):
        """
        Initializes the Cached instance. Loads data from the given path only if
        the instance is not already initialized.

        Args:
            path (str): The file path from which data is to be loaded.
        """
        if not hasattr(self, "_initialized"):
            if path == "":
                raise ValueError("path must not be empty")
            self.path = path
            if not callable(load_fn):
                raise ValueError("load_fn must be a callable function.")
            self.load_fn = load_fn
            self.data = self._load_data()
            self._initialized = True

    def _load_data(self):
        """
        Placeholder for loading data from the given path.

        Args:
            path (str): The file path from which data is to be loaded.

        Returns:
            str: Simulated data loaded from the file path. Replace with actual implementation.
        """
        return self.load_fn(self.path)

    def get_data(self):
        """
        Retrieves the data loaded from the file path.

        Returns:
            str: The loaded data.
        """
        return self.data


