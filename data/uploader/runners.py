"""Scripts to run when uploading"""

from baseball.data.chadwick.ids import uploader as chadwick
from baseball.data.pitch.savant import uploader as savant_pitch
from baseball.projects.strategery import run_expectancy as re24
from baseball.projects.strategery import win_probability as win_prob
from baseball.projects.strategery import win_probability_simulated as win_prob_simulated

RUNNERS = {
    "chadwick": chadwick.upload,
    "savant_pitch": savant_pitch.upload,
    "re24": re24.upload,
    "win_prob": win_prob.upload,
    "win_prob_simulated": win_prob_simulated.upload,
}
