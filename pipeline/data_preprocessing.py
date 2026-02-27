"""
LEASAR -- UR3 Real Data Preprocessing Pipeline
===============================================
Preprocesses raw UR3 robot sensor CSV logs into flat feature matrices
ready for anomaly detection model training.

Capabilities:
  - Parses list-encoded sensor columns (joint pos/vel/torque, TCP, forces)
  - Three acceleration backends: CPU (Numba JIT), GPU (CUDA kernel), Dask+cuDF
  - Kalman-style time-feature extraction
  - StandardScaler normalisation
  - Cycle segmentation by TCP home-position detection
  - Interactive 3D Plotly trajectory visualisation

Original research code: MTech190325.py
Author  : Harish Ramachandran
Inst    : SRM Institute of Science and Technology, Chennai (2025)
License : Apache 2.0
"""

import os
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Optional GPU backends
try:
    from numba import njit, prange, cuda
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cudf
    import dask_cudf as daskcudf
    CUDF_AVAILABLE = True
except ImportError:
    cudf = None
    daskcudf = None
    CUDF_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# -- Sensor column definitions -------------------------------------------------
LIST_COLUMNS = [
    "Actual Joint Positions",
    "Actual Joint Velocities",
    "Actual Joint Currents",
    "Actual Cartesian Coordinates",
    "Actual Tool Speed",
    "Generalized Forces",
    "Temperature of Each Joint",
    "Tool Acceleration",
    "Joint Voltages",
    "Elbow Position",
    "Elbow Velocity",
    "TCP Force",
]

# UR3 home position in Cartesian space (from real robot capture)
UR3_HOME_POS = (-0.132159, -0.2996955, 0.1653737)


# -- Utility functions ---------------------------------------------------------
def parse_float_list(value):
    """Safely parse a string-encoded list into a Python list of floats."""
    try:
        arr = ast.literal_eval(value)
        return [float(x) for x in arr]
    except Exception:
        return []


def expand_list_column(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Expand a list-valued column into numbered scalar columns."""
    df[colname] = df[colname].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    expanded = pd.DataFrame(df[colname].tolist(), index=df.index)
    expanded.columns = [f"{colname}{i+1}" for i in range(expanded.shape[1])]
    return expanded


# -- CPU backend (Numba JIT) ---------------------------------------------------
if NUMBA_AVAILABLE:
    def to_numba_list(py_list_of_lists):
        """Convert Python list-of-lists to Numba typed list."""
        typed_list = NumbaList()
        for row in py_list_of_lists:
            typed_sub = NumbaList()
            for val in row:
                typed_sub.append(val)
            typed_list.append(typed_sub)
        return typed_list

    @njit(parallel=True)
    def expand_lists_numba(numba_list_of_lists, max_length):
        """Numba-parallelised expansion of list data into 2D array."""
        n = len(numba_list_of_lists)
        result = np.zeros((n, max_length), dtype=np.float32)
        for i in prange(n):
            row = numba_list_of_lists[i]
            for j in range(len(row)):
                result[i, j] = row[j]
        return result

    def expand_column_cpu(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Expand list column using Numba JIT acceleration on CPU."""
        df[col] = df[col].apply(parse_float_list)
        max_length = df[col].apply(len).max()
        py_lists = df[col].tolist()
        numba_list = to_numba_list(py_lists)
        expanded = expand_lists_numba(numba_list, max_length)
        new_cols = {f"{col}{i+1}": expanded[:, i] for i in range(max_length)}
        df = df.drop(columns=[col])
        return df.assign(**new_cols)


# -- GPU backend (CUDA kernel) -------------------------------------------------
if NUMBA_AVAILABLE:
    @cuda.jit
    def expand_lists_kernel(numba_list_of_lists, max_length, result):
        """CUDA kernel: each thread expands one row."""
        row_idx = cuda.grid(1)
        if row_idx < result.shape[0]:
            for j in range(min(len(numba_list_of_lists[row_idx]), max_length)):
                result[row_idx, j] = numba_list_of_lists[row_idx][j]

    def expand_column_gpu(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Expand list column using CUDA GPU kernel."""
        df[col] = df[col].apply(parse_float_list)
        max_length = df[col].apply(len).max()
        py_lists = df[col].tolist()
        nrows = len(py_lists)
        numba_list = to_numba_list(py_lists)
        d_input = cuda.to_device(numba_list)
        d_result = cuda.device_array((nrows, max_length), dtype=np.float32)
        threads_per_block = 128
        blocks = (nrows + threads_per_block - 1) // threads_per_block
        expand_lists_kernel[blocks, threads_per_block](d_input, max_length, d_result)
        expanded = d_result.copy_to_host()
        new_cols = {f"{col}{i+1}": expanded[:, i] for i in range(max_length)}
        df = df.drop(columns=[col])
        return df.assign(**new_cols)


# -- Main preprocessing pipeline -----------------------------------------------
def preprocess(
    filepath: str,
    output_path: str = None,
    backend: str = "cpu",
    gpu_devices: str = "0",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for raw UR3 sensor CSV.

    Args:
        filepath    : Path to raw CSV (e.g. rightarmdropped.csv)
        output_path : If set, saves preprocessed CSV here
        backend     : "pandas" | "cpu" (Numba) | "gpu" (CUDA)
        gpu_devices : CUDA_VISIBLE_DEVICES string (e.g. "3,4")

    Returns:
        Preprocessed pd.DataFrame
    """
    if backend == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    # Extract time features from Timestamp if present
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"]   = df["Timestamp"].dt.hour
        df["minute"] = df["Timestamp"].dt.minute
        df["second"] = df["Timestamp"].dt.second
        df.drop("Timestamp", axis=1, inplace=True)

    # Expand list columns
    cols_present = [c for c in LIST_COLUMNS if c in df.columns]
    for col in cols_present:
        print(f"  Expanding column: {col}")
        if backend == "gpu" and NUMBA_AVAILABLE:
            df = expand_column_gpu(df, col)
        elif backend == "cpu" and NUMBA_AVAILABLE:
            df = expand_column_cpu(df, col)
        else:
            expanded = expand_list_column(df, col)
            df = pd.concat([df.drop(col, axis=1), expanded], axis=1)

    # Scale numerical features
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved preprocessed data to {output_path}")

    print(f"Preprocessing complete. Shape: {df.shape}")
    return df


# -- Cycle segmentation --------------------------------------------------------
def segment_cycles(
    df: pd.DataFrame,
    home_pos: tuple = UR3_HOME_POS,
    tolerance: float = 0.05,
) -> list:
    """
    Identify cycle boundaries based on when TCP returns to home position.

    Args:
        df        : DataFrame with Actual Cartesian Coordinates1/2/3 columns
        home_pos  : (x, y, z) home coordinates
        tolerance : Euclidean distance threshold to consider 'at home'

    Returns:
        List of (start_idx, end_idx) tuples for each detected cycle
    """
    coord_cols = [
        "Actual Cartesian Coordinates1",
        "Actual Cartesian Coordinates2",
        "Actual Cartesian Coordinates3",
    ]
    if not all(c in df.columns for c in coord_cols):
        raise ValueError("DataFrame must contain Cartesian coordinate columns.")

    coords = df[coord_cols].values
    home   = np.array(home_pos)
    dist   = np.linalg.norm(coords - home, axis=1)
    at_home = dist < tolerance

    cycles = []
    in_cycle = False
    start_idx = 0
    for i, h in enumerate(at_home):
        if not in_cycle and not h:
            start_idx = i
            in_cycle = True
        elif in_cycle and h:
            cycles.append((start_idx, i))
            in_cycle = False

    print(f"Detected {len(cycles)} cycles.")
    return cycles


# -- 3D Trajectory Visualisation -----------------------------------------------
def visualise_trajectory(
    df: pd.DataFrame,
    step_size: int = 100,
) -> None:
    """
    Create an animated 3D Plotly scatter plot of TCP trajectory.

    Args:
        df        : DataFrame with Actual Cartesian Coordinates1/2/3
        step_size : Number of points per animation frame
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed. Run: pip install plotly")
        return

    x_col = "Actual Cartesian Coordinates1"
    y_col = "Actual Cartesian Coordinates2"
    z_col = "Actual Cartesian Coordinates3"

    if not all(c in df.columns for c in [x_col, y_col, z_col]):
        print("Cartesian coordinate columns not found in DataFrame.")
        return

    frames = []
    for i in range(0, len(df), step_size):
        frames.append(go.Frame(
            data=go.Scatter3d(
                x=df[x_col].iloc[:i],
                y=df[y_col].iloc[:i],
                z=df[z_col].iloc[:i],
                mode="markers",
                marker=dict(size=3, colorscale="Viridis"),
            )
        ))

    fig = go.Figure(
        data=go.Scatter3d(
            x=df[x_col], y=df[y_col], z=df[z_col],
            mode="markers",
            marker=dict(size=3, colorscale="Viridis", showscale=True),
        ),
        frames=frames,
    )
    fig.update_layout(
        title="UR3 TCP 3D Trajectory",
        updatemenus=[{
            "buttons": [{"args": [None, {"frame": {"duration": 100, "redraw": True}}],
                         "label": "Play", "method": "animate"}]
        }],
    )
    fig.show()


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "rightarmdropped.csv"
    output   = sys.argv[2] if len(sys.argv) > 2 else "preprocessed.csv"
    backend  = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    df = preprocess(filepath, output_path=output, backend=backend)
    cycles = segment_cycles(df)
    print(f"First cycle: rows {cycles[0][0]} to {cycles[0][1]}" if cycles else "No cycles found.")
    visualise_trajectory(df)
