# System imports
import os
import sys

# Logging imports
import json
import logging

# Python imports
from typing import Callable, Tuple

# Math imports
import math
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


# TODO: Dark style
# TODO: Where's the legend?
# TODO: Colorbar?
# TODO: Votes bars come in sorted order always, instead of matching the names. Either animate the names or fix the bars' order.
class GovernmentPlotter:
    def __init__(
        self,
        registry: PersonRegistry,
        logger: logging.Logger,
        results_callback: Callable[
            [Candidate | None], Tuple[dict[str, int], np.ndarray]
        ],
    ):
        self.person_registry = registry
        self.logger = logger
        self.results_callback = results_callback

        self.fig, (self.ax_graph, self.ax_results) = plt.subplots(1, 2, figsize=(21, 9))

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        self.nearest_candidate: Candidate | None = None


        self.votes, heatmap = self.results_callback(self.nearest_candidate)
        self.results_bars = self.ax_results.bar(
            list(self.votes.keys()),
            list(self.votes.values()),
            color="forestgreen",
        )
        self.image = self.ax_graph.pcolormesh(heatmap, heatmap, heatmap, cmap="plasma")

        self.anim = FuncAnimation(
            self.fig,
            self.update_anim,
            init_func=self.init_anim,
            interval=16,
            repeat=False,
            blit=True,
        )

    def update_text_and_image(self):
        self.votes, heatmap = self.results_callback(self.nearest_candidate)
        for rect, height in zip(self.results_bars, list(self.votes.values())):
            rect.set_height(height)
        self.image.set_array(heatmap)

    def log(self, msg, level: int = logging.DEBUG):
        self.logger.log(level, msg)

    def on_click(self, event: MouseEvent):
        if event.button == 1 and event.inaxes == self.ax_graph:
            self.nearest_candidate = min(
                self.person_registry.get_candidates(),
                key=lambda candidate: candidate.distance(
                    Person([event.xdata, event.ydata])
                ),
            )

    def on_release(self, event: MouseEvent):
        if event.button == 1 and event.inaxes == self.ax_graph:
            self.update_text_and_image()
            self.nearest_candidate = None

    def on_motion(self, event: MouseEvent):
        if event.inaxes == self.ax_graph and self.nearest_candidate is not None:
            if event.xdata is not None and event.ydata is not None:
                self.nearest_candidate.values = [event.xdata, event.ydata]

    def plot(self):
        if config.NUM_AXES == 2:
            plt.show()

    def init_anim(self):
        self.ax_graph.set_xlabel("Apples or Mangoes")
        self.ax_graph.set_ylabel("Modern house or Spooky house")
        self.ax_graph.set_xlim(-1, 1)
        self.ax_graph.set_ylim(-1, 1)
        self.ax_graph.axvline(0, color="black")
        self.ax_graph.axhline(0, color="black")

        candidates_lines = self.ax_graph.plot(
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
            self.ax_graph.annotate(
                candidate.name,
                (
                    candidate.values[0],
                    candidate.values[1],
                ),
            )
            for candidate in self.person_registry.get_candidates()
        ]

        self.voters_lines = self.ax_graph.plot(
            [person.values[0] for person in self.person_registry.get_voters()],
            [person.values[1] for person in self.person_registry.get_voters()],
            "o",
            color="lightskyblue",
            label="Voters",
        )

        self.ax_graph.grid()
        handles, labels = self.ax_graph.get_legend_handles_labels()
        handles, labels = handles[-2:], labels[-2:]
        self.ax_graph.legend(handles, labels, loc="lower right")

        self.ax_results.set_xlabel("Candidates")
        self.ax_results.set_ylabel("Votes")
        self.ax_results.set_xticks(range(config.NUM_CANDIDATES))
        self.ax_results.set_xticklabels(
            [candidate.name for candidate in self.person_registry.get_candidates()]
        )
        for bar in self.results_bars:
            height = bar.get_height()
            self.ax_results.annotate(
                "{}".format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        # self.fig.colorbar(self.image)

        return (
            *candidates_lines,
            *self.voters_lines,
            *self.results_bars,
            *self.candidate_annotations,
            self.image,
        )

    def update_anim(self, i):
        candidates_lines = self.ax_graph.plot(
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

        return (
            *candidates_lines,
            *self.voters_lines,
            *self.results_bars,
            *self.candidate_annotations,
            self.image,
        )


# TODO: Thread pool parallelization
# TODO: Consider PyPy
# TODO: asyncio
# TODO: Real voter data
class Government:
    def __init__(self):
        heatmap_resolution = int(math.sqrt(config.NUM_VOTERS))
        self.heatmap = np.linspace(0, 1, num=config.NUM_VOTERS).reshape(
            heatmap_resolution, heatmap_resolution
        )

    def __enter__(self):
        self.person_registry = PersonRegistry()

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

    def simulate_voting(self, last_selected: Candidate | None):
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

        def map_range(x, in_min, in_max, out_min, out_max):
            return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

        if last_selected is not None:
            for iy, ix in np.ndindex(self.heatmap.shape):
                self.heatmap[iy, ix] = map_range(
                    last_selected.distance(
                        Person(
                            [
                                map_range(ix, 0, config.NUM_VOTERS, -1, 1),
                                map_range(iy, 0, config.NUM_VOTERS, -1, 1),
                            ]
                        )
                    ),
                    0,
                    math.sqrt(8),
                    0,
                    1,
                )

        return votes, self.heatmap

    def run(self):
        self.plotter.plot()


class GovernmentLogger(logging.Logger): 
    def __init__(self, govt_id):
        name = str(govt_id)

        super().__init__(name, config.LOG_LEVEL)

        if config.LOG_TO_TERMINAL:
            self.std_handler = logging.StreamHandler(sys.stdout)
            self.addHandler(self.std_handler)

        # Clean up the logs directory
        for f in [f for f in os.listdir("logs") if f.endswith(".log")]:
            os.remove(os.path.join("logs", f))

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
