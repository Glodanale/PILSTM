import subprocess
from variations import Variation

seeds = [25]
float_values = [32, 64]
lstm_units_list = [10, 100]
variations = [v.value for v in Variation]

entry_points = {
    "train_only": "train_only_exe.py",
    "train_eval": "train_validation_test_exe.py"
}

# Run both modes
for mode, entry_point in entry_points.items():
    for seed in seeds:
        for float_val in float_values:
            for units in lstm_units_list:
                for variation in variations:
                    print(f"Running {entry_point} | Seed={seed}, Float={float_val}, LSTM={units}, Variation={variation}")
                    subprocess.run([
                        "python", entry_point,
                        "--seed", str(seed),
                        "--float_value", str(float_val),
                        "--lstm_units", str(units),
                        "--variation", variation
                    ])
