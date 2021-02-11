
import sys


def get_size_recursive(obj):
    """
    Recursively finds the size of the given object/dict/iterable in bytes.
    :param obj: The object to find the size of
    :return: The size in bytes of obj (recursively including all objects referenced by obj)
    """
    return _get_size(obj, set())


def _get_size(obj, seen):
    size = sys.getsizeof(obj)
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(_get_size(v, seen) for v in obj.values())
        size += sum(_get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += _get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([_get_size(i, seen) for i in obj])

    return size
