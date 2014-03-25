__author__ = 'matt'

import correction
import features

from learning import CLASSIFIER_MAP


METHOD_MAP = {
    'correction': correction.METHOD_MAP,
    'features': features.METHOD_MAP
}


class APIBaseGenerator(object):

    API_DELIMITER = "."
    VERSION = 0.1

    def __init__(self, method_map):
        self._all_ptrs = []
        self._api = self.__generate_api_structure(method_map)
        self._method_map = method_map

    def __generate_api_structure(self, method_map):
        api = []

        def recursive_iterator(current_dict, prefix=''):

            for method_name in current_dict.keys():
                sub_branch = current_dict[method_name]
                identifier = self.API_DELIMITER.join([prefix, method_name])

                if callable(sub_branch):
                    api.append(identifier[1:])
                    self._all_ptrs.append(sub_branch)
                else:
                    recursive_iterator(sub_branch, identifier)

        recursive_iterator(method_map)

        return sorted(api)

    def is_valid(self, descriptor):
        """
        Returns Boolean based on whether the API method exists or not.
        """
        return descriptor in self._api

    def get_function_ptr(self, descriptor):
        """
        Accepts a string mapping resolution of module => function for use on frames
        or images (it is assumed all functions in this API are acting without extra
        parameters).

        Should variants with different parameters be required, make wrapper functions
        with associated docstrings denoting them.

        Iterates through the API tree to find the pointer to the required function,
        returns it.
        """
        if not self.is_valid(descriptor):
            return None

        api_components = descriptor.split(self.API_DELIMITER)
        api_components.reverse()  # Module => Function

        def traverse_api(api_stack, branch):
            this_part = api_stack.pop()
            found_block = branch[this_part]

            if callable(found_block):
                return found_block
            else:
                return traverse_api(api_stack, found_block)

        return traverse_api(api_components, self._method_map)

    @property
    def all_ptrs(self):
        return self._all_ptrs

    @property
    def structure(self):
        """
        Returns a list of strings for all methods and their nested locations.
        """
        return self._api


class APIFrameMetrics(APIBaseGenerator):
    """
    API Generator / Holder for all methods which may be used on images / frames (without
    extra parameters).
    """

    def __init__(self):
        super(APIFrameMetrics, self).__init__(method_map=METHOD_MAP)


class APIClassifiers(APIBaseGenerator):

    def __init__(self):
        super(APIClassifiers, self).__init__(method_map=CLASSIFIER_MAP)


class APIHolder(object):

    def __init__(self):
        self._api = {
            'methods': APIFrameMetrics(),
            'classifiers': APIClassifiers()
        }

    @property
    def everything(self):
        return self._api

    @property
    def methods(self):
        return self.everything['methods']

    @property
    def classifiers(self):
        return self.everything['classifiers']

    @property
    def version(self):
        return APIBaseGenerator.VERSION

    @property
    def structure(self):
        for api_type in self.everything.keys():
            api_object = self.everything[api_type]

            for pointer in api_object.structure:
                yield "[" + api_type + "] " +  pointer

_API = None


def get_api():
    global _API

    if _API == None:
        _API = APIHolder()

    return _API
