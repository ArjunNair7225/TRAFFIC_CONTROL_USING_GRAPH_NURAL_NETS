from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "pems-bay.h5"
ADJ_PATH = ROOT / "Data" / "adj_mx_bay.pkl"
MODEL_PATH = ROOT / "GCN" / "best_gcn_model.pth"
SEQ_LEN = 12
NUM_NODES = 325
CONGESTION_THRESHOLD = 50.0


class GCNLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.W = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.relu(adj @ self.W(x))


class GCN(torch.nn.Module):
    def __init__(self, seq_len: int, hidden: int = 64) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(seq_len, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return self.out(x).squeeze(-1)


def _find_speed_dataset(group: h5py.Group) -> np.ndarray | None:
    for name, item in group.items():
        if isinstance(item, h5py.Dataset) and item.ndim == 2 and item.shape[1] == NUM_NODES:
            return item[:].astype(np.float32)
        if isinstance(item, h5py.Group):
            found = _find_speed_dataset(item)
            if found is not None:
                return found
    return None


@st.cache_data(show_spinner=False)
def load_speed_data() -> np.ndarray:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    with h5py.File(DATA_PATH, "r") as file:
        speed = _find_speed_dataset(file)
    if speed is None:
        raise ValueError("Could not find a 2D speed dataset with 325 sensor columns.")
    return speed


def _extract_adjacency(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        arrays = [item for item in value if isinstance(item, np.ndarray)]
        for item in arrays:
            if item.shape == (NUM_NODES, NUM_NODES):
                return item
    raise ValueError("The adjacency file does not contain a 325 x 325 matrix.")


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[GCN, torch.Tensor]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    with ADJ_PATH.open("rb") as file:
        adjacency = _extract_adjacency(pickle.load(file, encoding="latin1")).astype(np.float32)
    adjacency = adjacency + np.eye(NUM_NODES, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized = inv_sqrt_degree @ adjacency @ inv_sqrt_degree

    model = GCN(SEQ_LEN)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, torch.from_numpy(normalized)


def predict(model: GCN, adjacency: torch.Tensor, history: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(history).astype(np.float32)
    inputs = torch.from_numpy(scaled).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(inputs, adjacency).squeeze(0).numpy()
    return scaler.inverse_transform(prediction.reshape(1, -1)).ravel()


def parse_manual_readings(value: str) -> np.ndarray:
    values = [float(item.strip()) for item in value.replace("\n", ",").split(",") if item.strip()]
    if len(values) != SEQ_LEN:
        raise ValueError(f"Please enter exactly {SEQ_LEN} readings, one for each 5-minute interval.")
    if any(item < 0 or item > 100 for item in values):
        raise ValueError("Readings should be realistic speeds between 0 and 100 mph.")
    return np.asarray(values, dtype=np.float32)


def readings_from_csv(uploaded_file: Any, sensor_number: int) -> np.ndarray:
    table = pd.read_csv(uploaded_file)
    numeric_columns = table.select_dtypes(include="number").columns.tolist()
    sensor_column = f"sensor_{sensor_number}"
    if sensor_column in table.columns:
        column = sensor_column
    elif len(numeric_columns) == 1:
        column = numeric_columns[0]
    elif len(numeric_columns) >= sensor_number:
        column = numeric_columns[sensor_number - 1]
    else:
        raise ValueError(
            f"Could not find readings for sensor {sensor_number}. "
            f"Use a '{sensor_column}' column or a CSV with one numeric speed column."
        )
    values = table[column].dropna().to_numpy(dtype=np.float32)
    if len(values) < SEQ_LEN:
        raise ValueError(f"Your CSV needs at least {SEQ_LEN} readings. It currently has {len(values)}.")
    values = values[-SEQ_LEN:]
    if np.any((values < 0) | (values > 100)):
        raise ValueError("CSV readings should be realistic speeds between 0 and 100 mph.")
    return values


def risk_label(speed: float) -> tuple[str, str]:
    if speed < 25:
        return "Critical", "#ff5b6e"
    if speed < CONGESTION_THRESHOLD:
        return "Congested", "#ffb454"
    return "Free flow", "#42d6a4"


def forecast_chart(history: np.ndarray, forecast: list[float]) -> go.Figure:
    input_values = history.tolist()
    all_values = input_values + forecast
    labels = [f"-{(SEQ_LEN - index) * 5} min" for index in range(SEQ_LEN)] + [
        f"+{(index + 1) * 5} min" for index in range(len(forecast))
    ]
    boundary = SEQ_LEN - 1
    figure = go.Figure()
    trace = go.Scatter(
        x=labels,
        y=all_values,
        mode="lines+markers",
        name="Traffic speed",
        line={"color": "#46d9ff", "width": 3},
        marker={"size": 8, "color": ["#46d9ff"] * SEQ_LEN + ["#ffb454"] * len(forecast)},
    )
    figure.add_trace(
        trace
    )
    figure.frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=labels[: index + 1],
                    y=all_values[: index + 1],
                    mode="lines+markers",
                    line={"color": "#46d9ff", "width": 3},
                    marker={"size": 8},
                )
            ],
            name=str(index),
        )
        for index in range(len(all_values))
    ]
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.13,
                "buttons": [
                    {
                        "label": "▶ Play forecast",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 180, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                            },
                        ],
                    }
                ],
            }
        ]
    )
    figure.add_trace(
        go.Scatter(x=[labels[-1]], y=[all_values[-1]], mode="markers", marker={"size": 1, "opacity": 0}, showlegend=False)
    )
    figure.add_vline(x=boundary + 0.5, line_dash="dash", line_color="#64748b")
    figure.add_annotation(
        x=boundary + 0.7,
        y=max(all_values) + 2,
        text="FORECAST",
        showarrow=False,
        font={"color": "#ffb454", "size": 11},
    )
    figure.update_layout(
        height=360,
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#b7c8d9"},
        xaxis={"showgrid": False},
        yaxis={"title": "Speed (mph)", "gridcolor": "rgba(148,163,184,.14)"},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    return figure


st.set_page_config(page_title="FlowSense | Traffic Intelligence", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#ecf7ff; --muted:#8ca5b9; --line:rgba(148,163,184,.18); }
    html, body, [class*="css"] { font-family:'Space Grotesk', sans-serif; }
    .stApp { background: radial-gradient(circle at 80% 0%, #123750 0%, #07131f 42%, #050a10 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#081522; border-right:1px solid var(--line); }
    .hero { padding: 22px 0 12px; }
    .eyebrow { color:#46d9ff; font-family:'DM Mono'; font-size:12px; letter-spacing:.16em; }
    h1 { font-size:44px !important; letter-spacing:-.05em; margin:5px 0 7px !important; }
    .subtitle { color:var(--muted); font-size:16px; max-width:740px; }
    .card { background:rgba(10,25,39,.72); border:1px solid var(--line); border-radius:18px; padding:18px; min-height:110px; }
    .metric-label { color:var(--muted); font-family:'DM Mono'; font-size:11px; text-transform:uppercase; letter-spacing:.09em; }
    .metric-value { color:var(--ink); font-size:30px; font-weight:700; margin-top:10px; }
    .metric-note { color:var(--muted); font-size:12px; margin-top:5px; }
    .status-pill { display:inline-block; padding:5px 10px; border-radius:99px; font-size:12px; font-weight:600; }
    .section-title { font-size:20px; font-weight:600; margin:15px 0 5px; }
    .stButton > button { border-radius:10px; border:1px solid #46d9ff; background:#46d9ff; color:#03111c; font-weight:700; }
    div[data-testid="stMetric"] { background:rgba(10,25,39,.72); border:1px solid var(--line); padding:15px; border-radius:16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="eyebrow">TRAFFIC INTELLIGENCE / PEMS-BAY</div>'
    '<h1>See the road ahead.</h1><div class="subtitle">A simple traffic demonstration: choose a road sensor, review the last hour, and see what the model expects to happen next.</div></div>',
    unsafe_allow_html=True,
)

try:
    speed_data = load_speed_data()
    model, adjacency = load_model()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(f"Unable to start the prediction engine: {error}")
    st.stop()

scaler = StandardScaler().fit(speed_data)
max_start = len(speed_data) - SEQ_LEN - 1
with st.sidebar:
    st.markdown("### Choose your data source")
    input_mode = st.radio(
        "How should the dashboard get readings?",
        ["Recorded demo data", "Enter readings myself", "Upload a CSV file"],
        help="Use recorded demo data for a replay. Use manual or CSV mode to test fresh readings from today.",
    )
    if input_mode == "Recorded demo data":
        st.caption("Replays the PEMS-BAY training-era data so you can demonstrate the project without an internet connection.")
    elif input_mode == "Enter readings myself":
        st.caption("Enter the latest 12 speed readings. Each reading should be 5 minutes apart.")
    else:
        st.caption("Upload a CSV containing at least 12 recent speed readings.")

    st.markdown("### Choose what to explore")
    sensor = st.number_input(
        "Road sensor number",
        min_value=1,
        max_value=NUM_NODES,
        value=1,
        step=1,
        help="Each sensor represents one location on the road network. The project has 325 sensors.",
    ) - 1
    start = max_start
    if input_mode == "Recorded demo data":
        start = st.slider(
            "Starting point in the dataset",
            0,
            max_start,
            max_start,
            help="The dataset is a timeline. Each step equals 5 minutes. Moving this slider lets you look at a different one-hour period.",
        )
    horizon = st.slider(
        "How far ahead should we look?",
        1,
        6,
        5,
        help="One forecast step means 5 minutes. For example, 5 steps means the model looks 25 minutes ahead.",
    )
    adjustment = 0.0
    if input_mode == "Recorded demo data":
        adjustment = st.slider(
            "Try a what-if speed change (mph)",
            -20.0,
            20.0,
            0.0,
            0.5,
            help="This is a demonstration control. It adds the chosen speed change to this sensor's last-hour readings before prediction. It does not change the original dataset.",
        )
    manual_text = ""
    uploaded_file = None
    if input_mode == "Enter readings myself":
        manual_text = st.text_area(
            "Latest 12 speeds (mph)",
            placeholder="Example: 62, 61, 60, 59, 57, 55, 54, 52, 50, 49, 48, 47",
            height=110,
            help="Enter exactly 12 speeds, separated by commas or new lines. The first is one hour ago; the last is the newest.",
        )
    elif input_mode == "Upload a CSV file":
        uploaded_file = st.file_uploader(
            "Upload recent readings",
            type=["csv"],
            help="Use a sensor_1, sensor_2, etc. column, or upload a CSV containing one numeric speed column.",
        )
    st.info("**In plain English:** the model reads the last 12 measurements (12 × 5 minutes = 1 hour), then predicts the next few measurements.")

base_window = speed_data[start : start + SEQ_LEN].copy()
input_error = None
input_source_note = "Using recorded PEMS-BAY demo readings."
input_ready = True
if input_mode == "Enter readings myself":
    try:
        live_readings = parse_manual_readings(manual_text) if manual_text.strip() else None
    except ValueError as error:
        live_readings = None
        input_error = str(error)
    if live_readings is not None:
        base_window[:, sensor] = live_readings
        input_source_note = f"Using your 12 manual readings for sensor {sensor + 1}."
    else:
        input_ready = False
elif input_mode == "Upload a CSV file":
    if uploaded_file is not None:
        try:
            live_readings = readings_from_csv(uploaded_file, sensor + 1)
            base_window[:, sensor] = live_readings
            input_source_note = f"Using the latest 12 readings from your CSV for sensor {sensor + 1}."
        except (ValueError, pd.errors.ParserError) as error:
            input_error = str(error)
            input_ready = False
    else:
        input_ready = False
edited_window = base_window.copy()
edited_window[:, sensor] += adjustment
history = edited_window[:, sensor]

st.markdown('<div class="section-title">Your traffic forecast</div>', unsafe_allow_html=True)
st.caption(input_source_note)
if input_error:
    st.warning(input_error)
last_speed = float(history[-1])
current_risk, risk_color = risk_label(last_speed)
st.caption(
    f"Sensor **{sensor + 1}** is currently showing **{last_speed:.1f} mph**. "
    "Press the button to ask the trained model what happens next."
)
if st.button("▶  Predict the next few minutes", width="stretch", disabled=not input_ready):
    st.session_state["run_prediction"] = True

if st.session_state.get("run_prediction", False):
    predictions: list[np.ndarray] = []
    rolling = edited_window.copy()
    for _ in range(horizon):
        next_speed = predict(model, adjacency, rolling, scaler)
        predictions.append(next_speed)
        rolling = np.vstack([rolling[1:], next_speed])
    forecast = [float(item[sensor]) for item in predictions]
else:
    forecast = [last_speed] * horizon

predicted_speed = forecast[-1]
predicted_risk, predicted_color = risk_label(predicted_speed)
delta = predicted_speed - last_speed
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="card"><div class="metric-label">What we see now</div><div class="metric-value">{last_speed:.1f} mph</div><div class="metric-note">Latest reading at road sensor {sensor + 1}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card"><div class="metric-label">Expected later</div><div class="metric-value">{predicted_speed:.1f} mph</div><div class="metric-note">{delta:+.1f} mph after {horizon * 5} minutes</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="card"><div class="metric-label">Road condition</div><div class="metric-value" style="color:{predicted_color}">{predicted_risk}</div><div class="metric-note">Below {CONGESTION_THRESHOLD:.0f} mph is treated as congestion</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="card"><div class="metric-label">Model track record</div><div class="metric-value">96.0%</div><div class="metric-note">Test accuracy reported for this project</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.55, 1])
with left:
    st.markdown('<div class="section-title">What the line means</div>', unsafe_allow_html=True)
    st.caption("Blue dots are real past readings. Orange dots are the model's prediction. The horizontal axis advances in 5-minute steps.")
    st.plotly_chart(forecast_chart(history, forecast), width="stretch", config={"displayModeBar": False})
with right:
    st.markdown('<div class="section-title">Where traffic may be slowest</div>', unsafe_allow_html=True)
    st.caption("These are the 10 sensor locations with the lowest predicted speeds. Lower bars mean slower traffic.")
    final_prediction = predictions[-1] if st.session_state.get("run_prediction", False) else speed_data[start + SEQ_LEN - 1]
    hotspot_indices = np.argsort(final_prediction)[:10]
    hotspot = go.Figure(
        go.Bar(
            x=final_prediction[hotspot_indices][::-1],
            y=[f"Sensor {index + 1}" for index in hotspot_indices[::-1]],
            orientation="h",
            marker_color=["#ff5b6e" if value < 25 else "#ffb454" for value in final_prediction[hotspot_indices][::-1]],
            text=[f"{value:.1f}" for value in final_prediction[hotspot_indices][::-1]],
            textposition="auto",
        )
    )
    hotspot.update_layout(height=360, margin={"l": 8, "r": 8, "t": 20, "b": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#b7c8d9"}, xaxis={"title": "mph", "gridcolor": "rgba(148,163,184,.14)"}, yaxis={"showgrid": False})
    st.plotly_chart(hotspot, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="section-title">The readings used by the prediction</div>', unsafe_allow_html=True)
st.caption(
    "The model does not use just one reading. It looks at 12 readings from the selected sensor: "
    "one reading every 5 minutes across the previous hour."
)
table = [
    {
        "When": f"{(SEQ_LEN - index) * 5} min ago" if index < SEQ_LEN - 1 else "Most recent",
        "Speed at sensor (mph)": round(float(base_window[index, sensor] + adjustment), 1),
    }
    for index in range(SEQ_LEN)
]
st.dataframe(table, width="stretch", hide_index=True)

with st.expander("What do the controls and graphs mean?"):
    st.markdown(
        f"""
        - **Recorded demo data:** Replays the original PEMS-BAY readings. This is useful when you want a reliable presentation without needing a live connection.
        - **Enter readings myself:** Lets you test the model with 12 fresh readings from a sensor. The readings should be 5 minutes apart.
        - **Upload a CSV file:** Lets you use a real export from your sensors. The file needs at least 12 rows and either a `sensor_{sensor + 1}` column or one numeric speed column.
        - **Road sensor number:** One of the {NUM_NODES} locations represented in the dataset. Changing it lets you inspect a different part of the road network.
        - **Starting point in the dataset:** Picks which moment in the recorded timeline to study. It is not a date; it is a position in the dataset. Every position is 5 minutes apart.
        - **How far ahead should we look?:** This is the **forecast horizon**. A value of 1 means 5 minutes ahead, 5 means 25 minutes ahead, and 6 means 30 minutes ahead.
        - **What-if speed change:** This is a safe classroom demonstration. A value of **-10 mph** asks, “What if this sensor became 10 mph slower?” A value of **+10 mph** asks, “What if it became 10 mph faster?”
        - **What we see now:** The most recent measured speed in the selected one-hour window.
        - **Expected later:** The final speed predicted at the end of your chosen forecast horizon.
        - **Road condition:** Free flow means 50 mph or more, congested means 25–49.9 mph, and critical means below 25 mph.
        - **Main line chart:** Blue is the past; orange is the future prediction. The vertical divider separates real readings from predictions. Use **Play forecast** to animate the timeline.
        - **Slowest locations chart:** A quick warning view showing which sensor locations may need attention first.

        **Important:** Manual and CSV modes replace the selected sensor's old readings with your fresh readings. The other sensor locations still use the latest available readings from the project dataset because the GCN expects all {NUM_NODES} sensors at each time step.
        """
    )
with st.expander("How does the AI make this prediction?"):
    st.markdown(
        "The project uses a **Graph Convolutional Network (GCN)**. In simple terms, it looks at the selected road and also learns from connected sensor locations. It combines those road relationships with the previous hour of speeds, then estimates the next speed in miles per hour. This is a demonstration using recorded PEMS-BAY data, not live GPS traffic."
    )
