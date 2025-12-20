import itertools

TRAIN_TEST_CUTOFF_DATE = "2024-12-31"


# very basic pitch shape features (think stuff)
SHAPE_FEATURES = [
    "release_speed",
    "arm_angle",
    "release_extension",
    "release_pos_x_batter_neutral",
    "release_pos_z",
    "pfx_x_batter_neutral",
    "pfx_z",
]

# location features
LOCATION_FEATURES = ["plate_x_batter_neutral", "plate_z"]

CONTEXT_FEATURES = [f"{i}_{j}" for i, j in itertools.product(range(4, 3))] + [
    "batter_ahead",
    "pitcher_ahead",
    "2K",
    "is_oppo_hand",
]

FEATURES = CONTEXT_FEATURES + SHAPE_FEATURES + LOCATION_FEATURES

CATEGORICAL_RESPONSE = "categorical_response_idx"
