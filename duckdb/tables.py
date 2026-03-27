import os
from types import SimpleNamespace

from baseball.data.chadwick.constants import CHADWICK_DUCK_DB_PARQUET_PATH
from baseball.data.savant.pitch.constants import SAVANT_DUCK_DB_PARQUET_PATH
from baseball.projects.pitch_quality.model.submodels.bayesian_fastball_differentials.constants import (
    SMOOTHED_FA_PLAYER_GAME_MEANS_DUCK_DB_PATH,
    SMOOTHED_FA_PLAYER_MEANS_DUCK_DB_PATH,
)
from baseball.projects.strategery.constants import (
    PE288_DUCK_DB_PARQUET_PATH,
    RE24_DUCK_DB_PARQUET_PATH,
    WIN_PROB_DUCK_DB_PARQUET_PATH,
)

# setting it u
SCHEMA = dict(
    # ---------------------------------- #
    # chadwick quasi-namespace
    # ---------------------------------- #
    chadwick=dict(ids=CHADWICK_DUCK_DB_PARQUET_PATH),
    # ---------------------------------- #
    # model outputs quasi-namespace
    # ---------------------------------- #
    model_outputs=dict(
        smoothed_fastball_shapes_player_season=SMOOTHED_FA_PLAYER_MEANS_DUCK_DB_PATH,
        smoothed_fastball_shapes_player_season_game=SMOOTHED_FA_PLAYER_GAME_MEANS_DUCK_DB_PATH,
    ),
    # ---------------------------------- #
    # strategery quasi-namespace
    # ---------------------------------- #
    strategery=dict(
        re24=RE24_DUCK_DB_PARQUET_PATH,
        pe288=PE288_DUCK_DB_PARQUET_PATH,
        win_probability=WIN_PROB_DUCK_DB_PARQUET_PATH,
    ),
    # ---------------------------------- #
    # pitch quasi-namespace
    # ---------------------------------- #
    pitch=dict(savant=SAVANT_DUCK_DB_PARQUET_PATH),
)


class DuckNamespaceContainer:
    def __init__(self) -> None:
        for namespace, tbl_configs in SCHEMA.items():
            self.add_attribute(
                namespace, SimpleNamespace(**{tbl: self.parse_filepath(path) for tbl, path in tbl_configs.items()})
            )

    def parse_filepath(self, file_path: str) -> str:
        """ """
        return (
            # if it's explicitly a single parquet (e.g. chadwick.ids), return that
            file_path
            if ".parquet" in file_path
            # otherwise tack on a star so that it searches all parquets
            # in the path (e.g. pitch.savant), which is usually how you want
            # to set things up if going seasonal or chunking files for storage.
            else os.path.join(file_path, "*.parquet")
        )

    def add_attribute(self, name, value):
        # name is a string, value is anything
        setattr(self, name, value)


TABLES = DuckNamespaceContainer()
