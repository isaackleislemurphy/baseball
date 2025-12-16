
import pandas as pd

from baseball.projects.pitch_quality.data.etl import load_pitch_quality_model_data

pitch_data_df = load_pitch_quality_model_data()
breakpoint()
print("complete")