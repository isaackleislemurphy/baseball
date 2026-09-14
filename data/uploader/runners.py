"""Scripts to run when uploading"""

from baseball.data.chadwick.ids import uploader as chadwick
from baseball.data.pitch.savant import uploader as savant_pitch

RUNNERS = {
    "chadwick": chadwick.main,
    "savant_pitch": savant_pitch.main,
}
