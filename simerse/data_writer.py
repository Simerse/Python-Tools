
from simerse.data_loader import make_dimension_list_getter


class writer:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        if not hasattr(owner, 'writers'):
            owner.writers = {}
            owner.dimensions = []
        owner.writers[name] = self.fn
        owner.dimensions.append(name)
        setattr(owner, name, self.fn)


class DataWriter:
    def __init__(self):
        self._num_observations = 0

    def __getattr__(self, item):
        if item == 'get_dimension_list':
            if not hasattr(type(self), 'get_dimension_list'):
                type(self).get_dimension_list = make_dimension_list_getter(type(self).dimensions)
            self.get_dimension_list = type(self).get_dimension_list
            return self.get_dimension_list
        raise AttributeError(f'Object {self} has no attribute {item}.')

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
        return self._num_observations

    def write(self, observations, dimensions='all'):
        if dimensions == 'all':
            dimensions = type(self).dimensions

        if isinstance(dimensions, (str, int)):
            dimensions = (dimensions,)
            observations = (observations,)

        try:
            num_new_observations = len(observations[0])
        except TypeError:
            observations = tuple((observation,) for observation in observations)
            num_new_observations = 1

        dimensions = self.get_dimension_list(dimensions)

        try:
            if len(dimensions) != len(observations):
                raise ValueError(f'Dimension list of length {len(dimensions)} has a different length from the '
                                 f'observation list of length {len(observations)}. Cannot write observations.')
        except (AttributeError, TypeError):
            raise TypeError('Given observations should be a parallel list with the dimension list (or a single'
                            ' observation if using only one dimension)')

        for d, (i, o) in zip(dimensions, enumerate(observations)):
            self.writers[d](self, i + self._num_observations, o)

        self._num_observations += num_new_observations
