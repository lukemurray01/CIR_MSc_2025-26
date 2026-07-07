# Colour scheme and general formatting for plotting throughout the thesis

METHOD_COLOURS = {
    "FTE": "#4477AA", # Blue
    "HH": "#228833", # Green
    "ProjEuler": "#AA3377", # Magenta
    "KL": "#EE6677", # Salmon
    "KLM": "#7B1FA2", # Purple
    "BLT": "#CC6677", # Rose
    "Exact": "#000000", # Black
}

METHOD_LINESTYLES = {
    "FTE" : "-",
    "HH": "--",
    "ProjEuler": "-.",
    "KL": ":",
    "KLM": "-",
    "BLT": "--",
    "Exact": "-"
}

REGIME_COLOURS = {
    "A": "#0D652D",  # green
    "B": "#174EA6",  # blue
    "C": "#E37400",  # orange
    "D": "#A50E0E",  # red
    "E": "#7B1FA2",  # purple
}

METHOD_LABELS = {
    "FTE": "Full Truncation Euler",
    "HH": "H-H Milstein",
    "ProjEuler": "Projected Euler",
    "KL": "Kelly-Lord",
    "KLM": "KLM backstopped adaptive",
    "BLT": "BLT splitting",
    "Exact": "Exact",
}

DEFAULT_LINEWIDTH = 1.8
EXACT_LINEWIDTH = 2.2
GRID_ALPHA = 0.25
