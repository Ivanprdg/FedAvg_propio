from __future__ import annotations
from pathlib import Path
import pandas as pd
from codecarbon import EmissionsTracker


def round_tracker(output_dir: Path, project_name: str) -> EmissionsTracker:
    return EmissionsTracker(project_name=project_name, output_dir=str(output_dir), save_to_file=True)


def clean_emissions_csv(output_dir: Path, project_name: str) -> None:
    import pandas as pd
    cols = [
        "timestamp","project_name","experiment_id","duration","emissions","emissions_rate",
        "cpu_power","gpu_power","ram_power","cpu_energy","gpu_energy","ram_energy",
        "energy_consumed","codecarbon_version","cpu_count","cpu_model","gpu_count",
        "ram_total_size","tracking_mode"
    ]
    for f in Path(output_dir).glob("*.csv"):
        if f.name.startswith(project_name):
            try:
                df = pd.read_csv(f)
                df = df[[c for c in cols if c in df.columns]]
                df.to_csv(f, index=False)
            except Exception:
                pass

def read_round_energy_wh(output_dir: Path, project_name: str) -> float:
    # CodeCarbon guarda kWh en la columna 'energy_consumed'
    for f in Path(output_dir).glob("*.csv"):
        if f.name.startswith(project_name):
            try:
                df = pd.read_csv(f)
                if len(df) and "energy_consumed" in df.columns:
                    kwh = float(df.iloc[-1]["energy_consumed"])
                    return kwh * 1000.0  # Wh
            except Exception:
                pass
    return 0.0
