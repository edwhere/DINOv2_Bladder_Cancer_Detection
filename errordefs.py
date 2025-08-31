"""Define error classes applicable to the DINO v2/v3 classifier project."""

class DirectoryNotFoundError(Exception):
    """Raise an exception if a directory does not exist."""
    pass

class EmptyPartitionsData(Exception):
    """Raise an exception if the partitions data directory is empty (no folder structure)."""
    pass

class UnknownStorageFormat(Exception):
    """Raise an error if the stored model does not include the 'type' field, which indicates
    adherence to the formats defined in this project."""
    pass

class UnknownFileExtension(Exception):
    """Raise an error if a file does not have the expected extension."""
    pass

class IncompatibleModel(Exception):
    """Raise an error if the model is not compatible with the data. For example, the reported labels
    for the data are different from stored labels in a model file."""
    pass