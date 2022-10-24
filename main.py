# System imports
import os
import sys

# Logging imports
import json
import logging
from typing import Callable

# Math imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.animation import FuncAnimation

# Miscellaneous imports
import names

# Local imports
import config


class Person:
    def __init__(self, values=None):
        self.values: list[float] = (
            np.random.uniform(-1, 1, config.NUM_AXES).tolist()
            if values is None
            else values
        )

    def distance(self, other: "Person"):
        return float(
            np.linalg.norm(
                [self.values[i] - other.values[i] for i in range(len(self.values))]
            )
        )


class Voter(Person):
    pass


class Candidate(Person):
    def __init__(self, values=None, name=None):
        super().__init__(values)
        self.name: str = names.get_full_name() if name is None else name


class PersonRegistry:
    def __init__(self):
        self.voters: list[Voter] = []
        self.candidates: list[Candidate] = []

        if config.GENERATE_REGISTRY_FROM_JSON:
            with open("logs/database.json", "r") as f:
                data: dict = json.load(f)

                for values in data["Voters"]:
                    self.add_voter(Voter(values))

                for name, values in dict(data["Candidates"]).items():
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

            dumpy["Voters"] = [voter.values for voter in self.voters]
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

    def get_candidate_by_name(self, name: str) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate

        return None


class GovernmentPlotter:
    def __init__(
        self,
        registry: PersonRegistry,
        logger: logging.Logger,
        results_callback: Callable[[], str],
    ):
        self.person_registry = registry
        self.logger = logger
        self.results_callback = results_callback

        self.fig, self.ax = plt.subplots(figsize=(10, 9))

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.nearest_candidate: Candidate | None = None

        self.anim = FuncAnimation(
            self.fig,
            self.update_anim,
            init_func=self.init_anim,
            interval=16,
            repeat=False,
            blit=True,
        )

        self.text = self.ax.text(-1, -1, self.results_callback(), fontsize=16)

    def log(self, msg, level: int = logging.DEBUG):
        self.logger.log(level, msg)

    def on_click(self, event: MouseEvent):
        if event.button == 1 and event.inaxes == self.ax:
            self.nearest_candidate = min(
                self.person_registry.get_candidates(),
                key=lambda candidate: candidate.distance(
                    Person([event.xdata, event.ydata])
                ),
            )

    def on_release(self, event: MouseEvent):
        if event.button == 1 and event.inaxes == self.ax:
            self.nearest_candidate = None
            self.text.set_text(self.results_callback())

    def on_motion(self, event: MouseEvent):
        if event.inaxes == self.ax and self.nearest_candidate is not None:
            if event.xdata is not None and event.ydata is not None:
                self.nearest_candidate.values = [event.xdata, event.ydata]

    def plot(self):
        if config.NUM_AXES == 2:
            plt.show()

    def init_anim(self):
        self.ax.set_xlabel("Apples or Mangoes")
        self.ax.set_ylabel("Modern house or Spooky house")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axvline(0, color="black")
        self.ax.axhline(0, color="black")

        candidates_lines = self.ax.plot(
            [
                candidate.values[0]
                for candidate in self.person_registry.get_candidates()
            ],
            [
                candidate.values[1]
                for candidate in self.person_registry.get_candidates()
            ],
            "o",
            color="darkorange",
            label="Candidates",
        )

        self.candidate_annotations = [
            self.ax.annotate(
                candidate.name,
                (
                    candidate.values[0],
                    candidate.values[1],
                ),
            )
            for candidate in self.person_registry.get_candidates()
        ]

        self.voters_lines = self.ax.plot(
            [person.values[0] for person in self.person_registry.get_voters()],
            [person.values[1] for person in self.person_registry.get_voters()],
            "o",
            color="lightskyblue",
            label="Voters",
        )

        plt.grid()
        handles, labels = self.ax.get_legend_handles_labels()
        handles, labels = handles[-2:], labels[-2:]
        plt.legend(handles, labels, loc="lower right")

        self.lines = candidates_lines + self.voters_lines

        return self.lines + self.candidate_annotations + [self.text]

    def update_anim(self, i):
        candidates_lines = self.ax.plot(
            [
                candidate.values[0]
                for candidate in self.person_registry.get_candidates()
            ],
            [
                candidate.values[1]
                for candidate in self.person_registry.get_candidates()
            ],
            "o",
            color="darkorange",
        )

        for annotation, candidate in zip(
            self.candidate_annotations, self.person_registry.get_candidates()
        ):
            annotation.set_position((candidate.values[0], candidate.values[1]))

        self.lines = candidates_lines + self.voters_lines

        return self.lines + self.candidate_annotations + [self.text]


class Government:
    def __enter__(self):
        self.person_registry = PersonRegistry()

        # Clean up the logs directory
        for f in [f for f in os.listdir("logs") if f.endswith(".log")]:
            os.remove(os.path.join("logs", f))

        self.logger = GovernmentLogger(id(self))

        self.plotter = GovernmentPlotter(
            self.person_registry, self.logger, self.simulate_voting
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()
        self.person_registry.write_to_json()  # dump the final db

    def log(self, msg, level: int = logging.DEBUG):
        self.logger.log(level, msg)

    def simulate_voting(self):
        votes = dict.fromkeys(self.person_registry.get_candidates(), 0)

        for voter in self.person_registry.get_voters():
            vote = min(
                self.person_registry.get_candidates(),
                key=lambda candidate: voter.distance(candidate),
            )
            votes[vote] += 1

        votes = dict(sorted(votes.items(), key=lambda x: x[1], reverse=True))
        votes = {candidate.name: votes[candidate] for candidate in votes}

        results = f"Votes: {json.dumps(votes, indent=4)}"

        self.log(results)

        return results

    def run(self):
        self.plotter.plot()


class GovernmentLogger(logging.Logger):
    def __init__(self, govt_id):
        name = str(govt_id)

        super().__init__(name, config.LOG_LEVEL)

        if config.LOG_TO_TERMINAL:
            self.std_handler = logging.StreamHandler(sys.stdout)
            self.addHandler(self.std_handler)

        self.file_handler = logging.FileHandler(f"logs/{name}.log")
        self.addHandler(self.file_handler)

        self.debug(f"Government Logger initialized for {name}\n\n")

    def close(self):
        for handler in self.handlers:
            self.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    with Government() as govt:
        govt.run()
