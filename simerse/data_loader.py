
def make_dimension_list_getter(possible_dimensions):
    possible_dimensions_list = list(possible_dimensions)
    possible_dimensions_set = set(possible_dimensions)

    def get_dimension_list(given):
        final_dimensions = []

        for given_dimension in given:
            try:
                to_add = possible_dimensions_list[int(given_dimension)]
                final_dimensions.append(to_add)
            except (TypeError, ValueError):
                if given_dimension not in possible_dimensions_set:
                    raise ValueError(f'Given dimension \'{given_dimension}\' is not a dimension (dimensions are'
                                     f' {possible_dimensions_list})')
                else:
                    final_dimensions.append(given_dimension)

        return final_dimensions

    return get_dimension_list


class loader:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        if not hasattr(owner, 'loaders'):
            owner.loaders = {}
            owner.dimensions = []
        owner.loaders[name] = self.fn
        owner.dimensions.append(name)
        setattr(owner, name, self.fn)


class cache:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, obj, typ=None):
        if not hasattr(typ, 'cache'):
            typ.cache = {}
        if self.getter.__name__ not in typ.cache:
            typ.cache[self.getter.__name__] = self.getter()
        return typ.cache[self.getter.__name__]

    def __set__(self, obj, val):
        raise AttributeError(f'Cached value {self.getter.__name__} is read-only.')

    def __delete__(self, obj):
        type(obj).cache.pop(self.getter.__name__)


class DataLoader:
    def __getattr__(self, name):
        if name == 'get_dimension_list':
            if not hasattr(type(self), 'get_dimension_list'):
                type(self).get_dimension_list = make_dimension_list_getter(type(self).dimensions)
            self.get_dimension_list = type(self).get_dimension_list
            return self.get_dimension_list
        raise AttributeError(f'Object {self} has no attribute {name}.')

    def summary(self):
        dimensions_string = ''
        # noinspection PyTypeChecker
        for dimension in self.dimensions:
            dimensions_string += f'\t{dimension}\n'
        return f"""
====== Summary of dataset: {self.name} ======

{self.description}

Dimensions: 
{dimensions_string}
License:
{self.license}
"""

    @property
    def dimensions(self):
        return type(self).dimensions

    @property
    def name(self):
        try:
            return type(self).name
        except AttributeError:
            return ""

    @property
    def description(self):
        try:
            return type(self).description
        except AttributeError:
            return ""

    @property
    def license(self):
        try:
            return type(self).license
        except AttributeError:
            return ""

    def __len__(self):
        try:
            return type(self).num_observations
        except AttributeError:
            return 'Length Unknown'

    def load(self, points, dimensions='all'):
        if dimensions == 'all':
            dimensions = type(self).dimensions

        was_one_dim = False
        if isinstance(dimensions, (str, int)):
            dimensions = (dimensions,)
            was_one_dim = True

        was_one_point = False
        if isinstance(points, int):
            points = (points,)
            was_one_point = True

        dimensions = self.get_dimension_list(dimensions)

        ret = tuple(self.loaders[d](self, points) for d in dimensions)

        if was_one_point:
            ret = tuple(d[0] for d in ret)

        if was_one_dim:
            ret = ret[0]

        return ret
