import numpy as np

OUT_DEGREE_LIM = (0, 100)
OUT_DEGREE_TICKS = np.linspace(0, 100, 5, dtype=int)

IN_DEGREE_LIM = (0, 100)
IN_DEGREE_TICKS = np.linspace(0, 100, 5, dtype=int)

OUT_DEGREE_COUNTS_LIM = (0, 120)
OUT_DEGREE_COUNTS_TICKS = np.linspace(0, 120, 3, dtype=int)

IN_DEGREE_COUNTS_LIM = (0, 120)
IN_DEGREE_COUNTS_TICKS = np.linspace(0, 120, 3, dtype=int)

IN_DEGREE_BINS = np.linspace(0, 55, 30, dtype=int)
OUT_DEGREE_BINS = np.linspace(0, 150, 30, dtype=int)

BIN_WIDTH = 4

FIG_SIZE = (11, 4)
MARKER_SIZE = 25

SUBPLOT_DIVISIONS = (1, 3)
AX0_LOCATION = (0, 0)
AX0_COLSPAN = 2
AX1_LOCATION = (0, 2)
AX1_COLSPAN = 1