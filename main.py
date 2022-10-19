# System imports
import os
import sys

# Python imports
from typing import Dict, List
import numpy.typing as npt
from enum import Enum

# Logging imports
import json
import logging

# Math imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import config


class PersonType(Enum):
    VOTER = 0
    CANDIDATE = 1


Person = npt.NDArray[np.float64]


class Government:
    def __init__(self):
        self.person_registry: Dict[PersonType, List[Person]] = dict()

        for person_type in PersonType:
            self.person_registry[person_type] = list()

        if config.GENERATE_REGISTRY_FROM_JSON:
            self.generate_registry_from_json()
        else:
            self.add_random(PersonType.CANDIDATE, 3)
            self.add_random(PersonType.VOTER, 100)
            self.write_registry_to_json()

        # Clean up the logs directory
        for f in [f for f in os.listdir("logs") if f.endswith(".log")]:
            os.remove(os.path.join("logs", f))

    def __enter__(self):
        self.logger = GovernmentLogger(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()

    def log(self, msg: str, level: int = logging.DEBUG):
        self.logger.log(level, msg)

    def write_registry_to_json(self):
        with open("logs/database.json", "w") as f:
            dumpy = dict()

            for person_type, values in self.person_registry.items():
                dumpy[person_type.name] = [person.tolist() for person in values]

            json.dump(dumpy, f, indent=4)

    def generate_registry_from_json(self):
        with open("logs/database.json", "r") as f:
            data: Dict[str, List[float]] = json.load(f)

            for person_type, value_lists in data.items():
                for values in value_lists:
                    self.person_registry[PersonType[person_type]].append(
                        np.array(values, dtype=np.float64)
                    )

    def add_random(self, person_type: PersonType, num: int):
        self.person_registry[person_type] = [
            np.random.uniform(-1, 1, size=config.NUM_AXES) for _ in range(num)
        ]

    def distance(self, a: Person, b: Person):
        return np.linalg.norm(a - b)

    def simulate_voting(self):
        for person_type, people in self.person_registry.items():
            self.log(f"{person_type}: {len(people)}")
            for person in people:
                self.log(f"\t{person}")

        # Plot the data
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.axvline(0, color="black")
        ax.axhline(0, color="black")

        for person_type, people in self.person_registry.items():
            x = [p[0] for p in people]
            y = [p[1] for p in people]
            ax.scatter(x, y, label=person_type.name)

        candidates = self.person_registry[PersonType.CANDIDATE]
        for voter in self.person_registry[PersonType.VOTER]:
            for candidate in candidates:
                self.log(f"{voter} -> {candidate} = {self.distance(voter, candidate)}")

        plt.grid()
        plt.legend()
        plt.show()

    def vote(self):

        return 0


class GovernmentLogger(logging.Logger):
    def __init__(self, govt: Government):
        name = str(id(govt))

        super().__init__(name, config.LOG_LEVEL)

        if config.LOG_TO_TERMINAL:
            self.std_handler = logging.StreamHandler(sys.stdout)
            self.addHandler(self.std_handler)

        self.file_handler = logging.FileHandler(f"logs/{name}.log")
        self.addHandler(self.file_handler)

        self.debug(f"Government Logger initialized for {name}\n\n")

    def close(self):
        handlers = self.handlers
        for handler in handlers:
            self.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    with Government() as govt:
        govt.simulate_voting()
