# System imports
import os
import sys

# Logging imports
import json
import logging

# Math imports
import numpy as np
import matplotlib.pyplot as plt

# Miscellaneous imports
import names

# Local imports
import config


class Person:
    def __init__(self, values=None, name=None):
        self.values: list[float] = (
            np.random.uniform(-1, 1, config.NUM_AXES).tolist()
            if values is None
            else values
        )
        self.name: str = names.get_full_name() if name is None else name

    def distance(self, other: "Person"):
        return float(
            np.linalg.norm(
                [self.values[i] - other.values[i] for i in range(len(self.values))]
            )
        )


class Voter(Person):
    pass


class Candidate(Person):
    pass


# FIXME: no duplicate named people
class PersonRegistry:
    def __init__(self):
        self.voters: list[Voter] = []
        self.candidates: list[Candidate] = []

        if config.GENERATE_REGISTRY_FROM_JSON:
            with open("logs/database.json", "r") as f:
                data: dict = json.load(f)

                for voter in data["Voters"]:
                    for name, values in voter.items():
                        self.add_voter(Voter(values, name))

                for candidate in data["Candidates"]:
                    for name, values in candidate.items():
                        self.add_candidate(Candidate(values, name))
        else:
            for _ in range(config.NUM_VOTERS):
                self.add_voter(Voter())

            for _ in range(config.NUM_CANDIDATES):
                self.add_candidate(Candidate())

            self.write_to_json()

    def write_to_json(self):
        with open("logs/database.json", "w") as f:
            dumpy = dict()

            dumpy["Voters"] = {voter.name: voter.values for voter in self.voters}
            dumpy["Candidates"] = {
                candidate.name: candidate.values for candidate in self.candidates
            }

            json.dump(dumpy, f, indent=4)

    def add_voter(self, voter: Voter):
        self.voters.append(voter)

    def add_candidate(self, candidate: Candidate):
        self.candidates.append(candidate)

    def get_voters(self) -> list[Voter]:
        return self.voters

    def get_candidates(self) -> list[Candidate]:
        return self.candidates

    def get_by_name(self) -> Person:
        return Person([0.0 for _ in range(config.NUM_AXES)], "BRUH")  # TODO


class Government:
    def __init__(self):
        self.person_registry = PersonRegistry()

        # Clean up the logs directory
        for f in [f for f in os.listdir("logs") if f.endswith(".log")]:
            os.remove(os.path.join("logs", f))

    def __enter__(self):
        self.logger = GovernmentLogger(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()

    def log(self, msg, level: int = logging.DEBUG):
        self.logger.log(level, msg)

    def simulate_voting(self):
        # for person_type, people in self.person_registry.items():
        # 	self.log(f"{person_type}: {len(people)}")
        # 	for person in people:
        # 		self.log(f"\t{person}")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel("Axis 1")
        ax.set_ylabel("Axis 2")
        ax.axvline(0, color="black")
        ax.axhline(0, color="black")

        ax.scatter(
            [person.values[0] for person in self.person_registry.get_voters()],
            [person.values[1] for person in self.person_registry.get_voters()],
            color="lightskyblue",
            label="Voters",
        )

        ax.scatter(
            [
                candidate.values[0]
                for candidate in self.person_registry.get_candidates()
            ],
            [
                candidate.values[1]
                for candidate in self.person_registry.get_candidates()
            ],
            color="darkorange",
            label="Candidates",
        )

        votes = dict.fromkeys(self.person_registry.get_candidates(), 0)

        for voter in self.person_registry.get_voters():
            vote = min(
                self.person_registry.get_candidates(),
                key=lambda candidate: voter.distance(candidate),
            )
            votes[vote] += 1

        votes = dict(sorted(votes.items(), key=lambda x: x[1], reverse=True))
        votes = {candidate.name: votes[candidate] for candidate in votes}

        self.log(f"Votes: {json.dumps(votes, indent=4)}")

        plt.grid()
        plt.legend()
        plt.show()


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
