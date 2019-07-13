from ..utils import RESOLUTION

class Base:
    """ A class that is declared for performing Interpolation.
    This class should not be called directly, use one of it's
    children.
    """
    def __init__(
        self,
        resolution = 'med',
        coordinate_types = 'Euclidean'
    ):
        self.resolution = RESOLUTION[resolution]
        self.coordinate_type = coordinate_types
