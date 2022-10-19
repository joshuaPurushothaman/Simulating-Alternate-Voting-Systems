import logging

LOG_LEVEL = logging.DEBUG

# Whether to log to the terminal
LOG_TO_TERMINAL = False

# Number of political issues to consider
NUM_AXES = 2 # 1 and 3 axes do not work right now :(

# This only works after you've ran it once with the default of False.
# When set to true, it will read in the data from the previous run.
GENERATE_REGISTRY_FROM_JSON = False
