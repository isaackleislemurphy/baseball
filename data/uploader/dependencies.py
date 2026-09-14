"""
Use this to define the DAG
"""

# fmt: off
# isort: off
DEPENDENCIES = {
    "chadwick": set(),
    "savant_pitch" : set(),
        "re24" : {"savant_pitch"},
        # "win_prob" : {"re24"},
            "win_prob_simulated" : {"re24"}
}
# isort: on
# fmt: on
