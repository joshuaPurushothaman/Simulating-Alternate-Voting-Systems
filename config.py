import logging

LOG_LEVEL = logging.DEBUG

# Whether to log to the terminal
LOG_TO_TERMINAL = True

# Number of political issues to consider
NUM_AXES = 2 # 1 and 3 axes do not work right now :(

# Number of voters and candidates
NUM_VOTERS = 512
NUM_CANDIDATES = 3

# This only works after you've ran it once with the default of False.
# When set to true, it will read in the data from the previous run.
GENERATE_REGISTRY_FROM_JSON = False
