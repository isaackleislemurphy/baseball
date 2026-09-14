"""Scripts to run when uploading"""

from baseball.data.chadwick.ids import uploader as chadwick
from baseball.data.pitch.savant import uploader as savant_pitch
from baseball.projects.strategery import run_expectancy as re24

RUNNERS = {"chadwick": chadwick.upload, "savant_pitch": savant_pitch.upload, "re24": re24.upload}
