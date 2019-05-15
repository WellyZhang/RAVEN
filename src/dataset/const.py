# -*- coding: utf-8 -*-


# Maximum number of components in a RPM
MAX_COMPONENTS = 2

# Canvas parameters
IMAGE_SIZE = 160
CENTER = (IMAGE_SIZE / 2, IMAGE_SIZE / 2)
DEFAULT_RADIUS = IMAGE_SIZE / 4
DEFAULT_WIDTH = 2

# Attribute parameters
# Number
NUM_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_MIN = 0
NUM_MAX = len(NUM_VALUES) - 1

# Uniformity
UNI_VALUES = [False, False, False, True]
UNI_MIN = 0
UNI_MAX = len(UNI_VALUES) - 1

# Type
TYPE_VALUES = ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
TYPE_MIN = 0
TYPE_MAX = len(TYPE_VALUES) - 1

# Size
SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIZE_MIN = 0
SIZE_MAX = len(SIZE_VALUES) - 1

# Color
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
COLOR_MIN = 0
COLOR_MAX = len(COLOR_VALUES) - 1

# Angle: self-rotation
ANGLE_VALUES = [-135, -90, -45, 0, 45, 90, 135, 180]
ANGLE_MIN = 0
ANGLE_MAX = len(ANGLE_VALUES) - 1

META_TARGET_FORMAT = ["Constant", "Progression", "Arithmetic", "Distribute_Three", "Number", "Position", "Type", "Size", "Color"]
META_STRUCTURE_FORMAT = ["Singleton", "Left_Right", "Up_Down", "Out_In", "Left", "Right", "Up", "Down", "Out", "In", "Grid", "Center_Single", "Distribute_Four", "Distribute_Nine", "Left_Center_Single", "Right_Center_Single", "Up_Center_Single", "Down_Center_Single", "Out_Center_Single", "In_Center_Single", "In_Distribute_Four"]

# Rule, Attr, Param
# The design encodes rule priority order: Number/Position always comes first
# Number and Position could not both be sampled
# Progression on Number: Number on each Panel +1/2 or -1/2
# Progression on Position: Entities on each Panel roll over the layout
# Arithmetic on Number: Numeber on the third Panel = Number on first +/- Number on second (1 for + and -1 for -)
# Arithmetic on Position: 1 for SET_UNION and -1 for SET_DIFF
# Distribute_Three on Number: Three numbers through each row
# Distribute_Three on Position: Three positions (same number) through each row
# Constant on Number/Position: Nothing changes
# Progression on Type: Type progression defined as the number of edges on each entity (Triangle, Square, Pentagon, Hexagon, Circle)
# Distribute_Three on Type: Three types through each row
# Constant on Type: Nothing changes
# Progression on Size: Size on each entity +1/2 or -1/2
# Arithmetic on Size: Size on the third Panel = Size on the first +/- Size on the second (1 for + and -1 for -)
# Distribute_Three on Size: Three sizes through each row
# Constant on Size: Nothing changes
# Progression on Color: Color +1/2 or -1/2
# Arithmetic on Color: Color on the third Panel = Color on the first +/- Color on the second (1 for + and -1 for -)
# Distribute_Three on Color: Three colors through each row
# Constant on Color: Nothing changes
# Note that all rules on Type, Size and Color enforce value consistency in a panel
RULE_ATTR = [[["Progression", "Number", [-2, -1, 1, 2]], 
              ["Progression", "Position", [-2, -1, 1, 2]], 
              ["Arithmetic", "Number", [1, -1]],
              ["Arithmetic", "Position", [1, -1]],
              ["Distribute_Three", "Number", None],
              ["Distribute_Three", "Position", None],
              ["Constant", "Number/Position", None]],
             [["Progression", "Type", [-2, -1, 1, 2]],
              ["Distribute_Three", "Type", None], 
              ["Constant", "Type", None]],
             [["Progression", "Size", [-2, -1, 1, 2]],
              ["Arithmetic", "Size", [1, -1]],
              ["Distribute_Three", "Size", None],
              ["Constant", "Size", None]],
             [["Progression", "Color", [-2, -1, 1, 2]],
              ["Arithmetic", "Color", [1, -1]],
              ["Distribute_Three", "Color", None],
              ["Constant", "Color", None]]]
