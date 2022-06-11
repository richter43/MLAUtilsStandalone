import enum

#labels for each cancer class
class RenalCancerType(enum.Enum):
    NOT_CANCER = 0
    ONCOCYTOMA = 1
    CHROMOPHOBE = 2
    CLEAR_CELL = 3
    PAPILLARY = 4