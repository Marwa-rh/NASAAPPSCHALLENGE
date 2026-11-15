"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AERIS — Astronaut Environmental & Real-time Intelligence System     ║
║          Spaceflight Bio-Vitals & Astronaut Health Monitoring Backend        ║
║          FastAPI  |  WebSocket  |  Pandas  |  NumPy  |  Pydantic            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("AERIS")

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Scenario Definitions
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(__file__), "astronaut_vitals.csv")
INDEX_PATH = Path(__file__).parent / "public" / "index.html"

STORE_FORWARD_MAX = 500          # max packets in deep-space buffer
TELEMETRY_HZ      = 1.0         # tick rate (seconds)

# Alert codes
ALERT_HYPOXIA       = "EM-MED-804"
ALERT_SOLAR_FLARE   = "EM-RAD-901"

# Scenario envelopes — each key defines deltas/multipliers applied on top of
# the astronaut's personal baseline.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "NOMINAL": {
        "label":               "Nominal Spaceflight Operations",
        "co2_offset":          0,
        "co2_spike":           False,
        "spo2_offset":         0,
        "radiation_multiplier": 1.0,
        "hr_offset":           0,
        "temp_offset":         0.0,
    },
    "HYPOXIA": {
        "label":               "Suit Leak / Hypoxia Emergency",
        "co2_offset":          1300,       # pushes CO2 to ~2500 PPM
        "co2_spike":           True,
        "spo2_offset":         -10,        # drops SpO2 below 85%
        "radiation_multiplier": 1.0,
        "hr_offset":           25,
        "temp_offset":         0.4,
    },
    "SOLAR_FLARE": {
        "label":               "Solar Particle Event — Radiation Flare",
        "co2_offset":          0,
        "co2_spike":           False,
        "spo2_offset":         0,
        "radiation_multiplier": 3.5,       # pushes radiation rate to ~1.5 mSv/hr
        "hr_offset":           15,
        "temp_offset":         0.2,
    },
    "CARDIAC_STRESS": {
        "label":               "Microgravity Cardiovascular Stress Event",
        "co2_offset":          200,
        "co2_spike":           False,
        "spo2_offset":         -3,
        "radiation_multiplier": 1.1,
        "hr_offset":           55,         # sustained elevation > threshold
        "temp_offset":         0.6,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

class AERISState:
    """Central mutable state for the AERIS backend (singleton)."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.current_scenario: str = "NOMINAL"
        self.store_forward_queue: deque = deque(maxlen=STORE_FORWARD_MAX)
        self.mission_elapsed_seconds: float = 0.0
        self.boot_time: datetime = datetime.now(timezone.utc)
        self.active_websockets: int = 0

    # ── Convenience accessors ──────────────────────────────────────────────

    @property
    def scenario_config(self) -> Dict[str, Any]:
        return SCENARIOS[self.current_scenario]

    def get_astronaut(self, astronaut_id: int) -> Optional[pd.Series]:
        if self.df is None:
            return None
        row = self.df[self.df["Astronaut_ID"] == astronaut_id]
        return row.iloc[0] if not row.empty else None

    def set_scenario(self, preset: str) -> None:
        preset = preset.upper()
        if preset not in SCENARIOS:
            raise ValueError(f"Unknown scenario preset '{preset}'. "
                             f"Valid: {list(SCENARIOS.keys())}")
        self.current_scenario = preset
        logger.info("AERIS scenario switched → %s (%s)",
                    preset, SCENARIOS[preset]["label"])


state = AERISState()

# ─────────────────────────────────────────────────────────────────────────────
# CSV Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv() -> None:
    """Load and validate the astronaut vitals CSV into the global state."""
    if not os.path.exists(CSV_PATH):
        logger.warning(
            "CSV not found at '%s'. Endpoints requiring astronaut data will "
            "return 503 until the file is available.", CSV_PATH
        )
        state.df = None
        return

    try:
        df = pd.read_csv(CSV_PATH)

        # Normalise column names (strip stray whitespace)
        df.columns = [c.strip() for c in df.columns]

        required = {
            "Astronaut_ID", "Name", "Age_at_Mission_Start", "Gender",
            "Country", "Mission_Duration (days)", "Number_of_Missions",
            "Bone_Density_Loss (%)", "Muscle_Mass_Loss (%)",
            "Radiation_Exposure (mSv)", "Cardiovascular_Risk_Score",
            "Immune_Response_Suppression (%)", "Psychological_Impact_Score",
            "Vision_Change", "Countermeasures_Used",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        df["Astronaut_ID"] = df["Astronaut_ID"].astype(int)
        state.df = df
        logger.info("Loaded %d astronaut baseline records from '%s'.",
                    len(df), CSV_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load CSV: %s", exc)
        state.df = None


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

def _sine(t: float, period: float, amplitude: float, phase: float = 0.0) -> float:
    """Sinusoidal component for biological variability."""
    return amplitude * math.sin(2 * math.pi * t / period + phase)


def _noise(scale: float) -> float:
    """Gaussian noise helper."""
    return float(np.random.normal(0, scale))


def _calibrate_baseline(row: Optional[pd.Series]) -> Dict[str, float]:
    """
    Derive per-astronaut telemetry baselines from CSV data.
    Higher cardiovascular risk → elevated resting HR.
    Higher radiation exposure history → higher ambient radiation rate.
    Higher muscle-mass loss → lower exercise tolerance (lower SpO2 floor).
    """
    if row is None:
        return {
            "base_hr":             70.0,
            "base_spo2":           97.0,
            "base_temp":           37.0,
            "base_co2":            600.0,
            "base_rad_rate":       0.18,
            "base_cumulative_rad": 0.0,
            "base_pressure":       14.7,
            "cardio_risk":         5,
        }

    cardio  = float(row["Cardiovascular_Risk_Score"])           # 1–10
    rad_exp = float(row["Radiation_Exposure (mSv)"])            # 84–200
    muscle  = float(row["Muscle_Mass_Loss (%)"])                # 1.0–3.0

    # Heart Rate: scale from 68 bpm (low risk) → 82 bpm (high risk)
    base_hr = 68.0 + (cardio / 10.0) * 14.0

    # SpO2: slight reduction for high muscle-mass loss (deconditioning)
    base_spo2 = 98.0 - (muscle - 1.0) * 0.5

    # Ambient radiation rate: normalise historical exposure → 0.10–0.45 mSv/hr
    base_rad_rate = 0.10 + ((rad_exp - 84) / (200 - 84)) * 0.35

    return {
        "base_hr":             base_hr,
        "base_spo2":           min(base_spo2, 99.5),
        "base_temp":           37.0,
        "base_co2":            600.0,
        "base_rad_rate":       base_rad_rate,
        "base_cumulative_rad": rad_exp,       # carry historical dose forward
        "base_pressure":       14.7,
        "cardio_risk":         int(cardio),
    }


def generate_telemetry_tick(
    astronaut_id: int,
    t: float,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """
    Synthesise one telemetry tick.

    Parameters
    ----------
    astronaut_id      : requested astronaut (1-indexed)
    t                 : wall-clock time (seconds since epoch) for reproducibility
    elapsed_seconds   : mission elapsed time for cumulative radiation
    """
    row      = state.get_astronaut(astronaut_id)
    cal      = _calibrate_baseline(row)
    scenario = state.scenario_config

    # ── Vitals ────────────────────────────────────────────────────────────────
    hr = (
        cal["base_hr"]
        + _sine(t, 10, 3.0)           # respiratory sinus arrhythmia (10 s)
        + _sine(t, 90, 5.0, 1.2)      # slow autonomic wave (90 s)
        + _noise(1.5)
        + scenario["hr_offset"]
    )
    hr = round(max(45.0, min(hr, 220.0)), 1)

    spo2 = (
        cal["base_spo2"]
        + _sine(t, 25, 0.4)
        + _noise(0.3)
        + scenario["spo2_offset"]
    )
    spo2 = round(max(70.0, min(spo2, 100.0)), 1)

    core_temp = (
        cal["base_temp"]
        + _sine(t, 86400, 0.5, math.pi)  # circadian rhythm (24 h)
        + _sine(t, 120, 0.1)
        + _noise(0.05)
        + scenario["temp_offset"]
    )
    core_temp = round(max(35.0, min(core_temp, 41.0)), 2)

    # ── Suit Environment ──────────────────────────────────────────────────────
    if scenario["co2_spike"]:
        # Hypoxia scenario: CO2 climbs steeply, then oscillates at danger level
        co2 = (
            2500.0
            + _sine(t, 60, 150.0)
            + _noise(60.0)
            + scenario["co2_offset"]
        )
    else:
        co2 = (
            cal["base_co2"]
            + _sine(t, 300, 80.0)
            + _noise(25.0)
            + scenario["co2_offset"]
        )
    co2 = round(max(200.0, co2), 1)

    rad_rate = (
        cal["base_rad_rate"]
        + _sine(t, 600, 0.03)
        + _noise(0.008)
    ) * scenario["radiation_multiplier"]
    rad_rate = round(max(0.0, rad_rate), 4)

    # Cumulative dose: baseline historical + mission elapsed accumulation
    cumulative_rad = round(
        cal["base_cumulative_rad"] + (elapsed_seconds / 3600.0) * rad_rate, 3
    )

    suit_pressure = (
        14.7
        + _sine(t, 200, 0.05)
        + _noise(0.01)
    )
    suit_pressure = round(max(13.5, min(suit_pressure, 15.5)), 3)

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    alerts = _run_anomaly_checks(
        spo2       = spo2,
        co2        = co2,
        rad_rate   = rad_rate,
        heart_rate = hr,
        cardio_risk= cal["cardio_risk"],
    )

    # ── Assemble Packet ───────────────────────────────────────────────────────
    packet: Dict[str, Any] = {
        "aeris_version":   "2.0.0",
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "mission_elapsed": round(elapsed_seconds, 1),
        "astronaut_id":    astronaut_id,
        "scenario":        state.current_scenario,
        "vitals": {
            "heart_rate":  hr,
            "spo2":        spo2,
            "core_temp":   core_temp,
        },
        "suit_env": {
            "co2_level_ppm":          co2,
            "radiation_rate_mSv_hr":  rad_rate,
            "cumulative_radiation_mSv": cumulative_rad,
            "suit_pressure_psi":      suit_pressure,
        },
        "alerts": alerts,
        "alert_count": len(alerts),
    }

    return packet


def _run_anomaly_checks(
    spo2:        float,
    co2:         float,
    rad_rate:    float,
    heart_rate:  float,
    cardio_risk: int,
) -> List[Dict[str, Any]]:
    """
    Rule-based medical anomaly detection engine.

    Rules
    -----
    1. EM-MED-804 (CRITICAL) : SpO2 < 90 % OR CO2 > 1800 PPM
    2. EM-RAD-901 (CRITICAL) : Radiation Rate > 0.8 mSv/hr
    3. (unnamed)  (WARNING)  : Heart Rate > (140 - CardioRiskScore × 3)
    """
    alerts: List[Dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()

    # Rule 1 — Hypoxia / CO2 scrubber failure
    if spo2 < 90.0 or co2 > 1800.0:
        triggers = []
        if spo2 < 90.0:
            triggers.append(f"SpO2={spo2}% (threshold <90%)")
        if co2 > 1800.0:
            triggers.append(f"CO2={co2} PPM (threshold >1800 PPM)")
        alerts.append({
            "code":      ALERT_HYPOXIA,
            "severity":  "CRITICAL",
            "category":  "MEDICAL",
            "message":   "Hypoxia or CO2 Scrubber Anomaly Detected",
            "triggers":  triggers,
            "timestamp": ts,
            "action":    "IMMEDIATE EVA ABORT — Activate emergency O2 supply",
        })

    # Rule 2 — Solar Particle Event
    if rad_rate > 0.8:
        alerts.append({
            "code":      ALERT_SOLAR_FLARE,
            "severity":  "CRITICAL",
            "category":  "RADIATION",
            "message":   "Solar Radiation Flare / SPE Threshold Exceeded",
            "triggers":  [f"Radiation Rate={rad_rate} mSv/hr (threshold >0.8 mSv/hr)"],
            "timestamp": ts,
            "action":    "CREW SHELTER IN SHIELDED MODULE IMMEDIATELY",
        })

    # Rule 3 — Cardiovascular stress
    hr_threshold = 140 - (cardio_risk * 3)
    if heart_rate > hr_threshold:
        alerts.append({
            "code":      "EM-CARD-305",
            "severity":  "WARNING",
            "category":  "CARDIOVASCULAR",
            "message":   "Elevated Heart Rate — Cardiovascular Stress Detected",
            "triggers":  [
                f"HR={heart_rate} BPM (threshold >{hr_threshold} BPM "
                f"for CardioRiskScore={cardio_risk})"
            ],
            "timestamp": ts,
            "action":    "Reduce physical activity — Flight Surgeon notification required",
        })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    preset: str


class ScenarioResponse(BaseModel):
    status:           str
    previous_scenario: str
    new_scenario:     str
    label:            str
    message:          str


class SystemStatusResponse(BaseModel):
    system:          str
    version:         str
    status:          str
    boot_time_utc:   str
    current_time_utc: str
    current_scenario: str
    scenario_label:  str
    astronauts_loaded: int
    buffer_packets:  int
    active_streams:  int
    endpoints:       Dict[str, str]


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

# ── Lifespan Context Manager ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("  AERIS v2.0.0  —  Spaceflight Bio-Vitals Monitoring System")
    logger.info("=" * 70)
    load_csv()
    logger.info("Scenario:  %s — %s", state.current_scenario, SCENARIOS[state.current_scenario]["label"])
    logger.info("Store-and-Forward buffer capacity: %d packets", STORE_FORWARD_MAX)
    logger.info("AERIS backend ready. 🚀")
    yield
    logger.info("AERIS backend shutting down gracefully.")

app = FastAPI(
    title       = "AERIS — Astronaut Environmental & Real-time Intelligence System",
    description = (
        "Production-grade FastAPI backend for Spaceflight Bio-Vitals & "
        "Astronaut Health Monitoring."
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
# ─────────────────────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/api/status",
    response_model=SystemStatusResponse,
    summary="AERIS System Status Overview",
    tags=["System"],
)
async def system_status() -> SystemStatusResponse:
    """Returns live system health, current scenario, and endpoint manifest."""
    astronauts_loaded = len(state.df) if state.df is not None else 0
    return SystemStatusResponse(
        system           = "AERIS — Astronaut Environmental & Real-time Intelligence System",
        version          = "2.0.0",
        status           = "OPERATIONAL" if state.df is not None else "DEGRADED (CSV missing)",
        boot_time_utc    = state.boot_time.isoformat(),
        current_time_utc = datetime.now(timezone.utc).isoformat(),
        current_scenario = state.current_scenario,
        scenario_label   = state.scenario_config["label"],
        astronauts_loaded= astronauts_loaded,
        buffer_packets   = len(state.store_forward_queue),
        active_streams   = state.active_websockets,
        endpoints        = {
            "GET  /":                      "AERIS monitoring dashboard",
            "GET  /api/status":            "System status",
            "GET  /api/astronauts":        "List all astronaut baseline profiles",
            "GET  /api/astronauts/{id}":   "Single astronaut baseline profile",
            "POST /api/scenario":          "Switch hackathon demo scenario preset",
            "GET  /api/queue":             "Deep-space store-and-forward buffer",
            "WS   /ws/telemetry":          "1 Hz real-time telemetry stream (?astronaut_id=N)",
            "GET  /docs":                  "OpenAPI interactive documentation",
            "GET  /redoc":                 "ReDoc API documentation",
        },
    )


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get(
    "/api/astronauts",
    summary="List All Astronaut Baseline Profiles",
    tags=["Astronauts"],
)
async def list_astronauts() -> Dict[str, Any]:
    """Returns all 75 astronaut baseline records loaded from CSV."""
    _assert_csv_loaded()
    records = state.df.where(pd.notnull(state.df), None).to_dict(orient="records")
    return {
        "total":      len(records),
        "astronauts": records,
    }


@app.get(
    "/api/astronauts/{astronaut_id}",
    summary="Get Single Astronaut Baseline Profile",
    tags=["Astronauts"],
)
async def get_astronaut(astronaut_id: int) -> Dict[str, Any]:
    """
    Returns the baseline health profile and calibrated telemetry parameters
    for a specific astronaut by their numeric ID (1–75).
    """
    _assert_csv_loaded()
    row = state.get_astronaut(astronaut_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Astronaut with ID {astronaut_id} not found. Valid range: 1–75.",
        )
    cal = _calibrate_baseline(row)
    profile = {k: (None if pd.isna(v) else v)
               for k, v in row.to_dict().items()}
    return {
        "profile":               profile,
        "calibrated_telemetry":  cal,
        "hr_stress_threshold":   140 - cal["cardio_risk"] * 3,
        "scenario":              state.current_scenario,
    }


@app.post(
    "/api/scenario",
    response_model=ScenarioResponse,
    summary="Switch Hackathon Demo Scenario Preset",
    tags=["Scenario Control"],
)
async def set_scenario(request: ScenarioRequest) -> ScenarioResponse:
    """
    Switches the global telemetry scenario for all active WebSocket streams.

    Valid presets:
    - **NOMINAL** — Normal spaceflight operations
    - **HYPOXIA** — Suit leak: CO2 spikes, SpO2 drops
    - **SOLAR_FLARE** — Solar Particle Event: radiation spikes
    - **CARDIAC_STRESS** — Sustained cardiovascular effort
    """
    previous = state.current_scenario
    try:
        state.set_scenario(request.preset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ScenarioResponse(
        status            = "SUCCESS",
        previous_scenario = previous,
        new_scenario      = state.current_scenario,
        label             = state.scenario_config["label"],
        message           = (
            f"Global scenario switched from {previous} → {state.current_scenario}. "
            "All active WebSocket streams will reflect the new scenario on the next tick."
        ),
    )


@app.get(
    "/api/queue",
    summary="Deep-Space Store-and-Forward Buffer",
    tags=["Telemetry Buffer"],
)
async def get_queue(
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Returns packets stored in the deep-space latency buffer (last 500 packets).
    Simulates offline buffering when Earth communication links are unavailable.

    - **limit** : number of packets to return (default 50, max 500)
    - **offset**: pagination offset into the queue
    """
    limit  = max(1, min(limit, STORE_FORWARD_MAX))
    offset = max(0, offset)
    packets = list(state.store_forward_queue)
    total   = len(packets)
    page    = packets[offset: offset + limit]
    return {
        "buffer_capacity":    STORE_FORWARD_MAX,
        "total_buffered":     total,
        "returned":           len(page),
        "offset":             offset,
        "limit":              limit,
        "oldest_packet_utc":  packets[0]["timestamp_utc"] if packets else None,
        "newest_packet_utc":  packets[-1]["timestamp_utc"] if packets else None,
        "packets":            page,
    }

# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Real-Time Telemetry Stream
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def websocket_telemetry(
    websocket: WebSocket,
    astronaut_id: int = 1,
) -> None:
    """
    Real-time 1 Hz telemetry WebSocket stream.

    Connect with:  ws://host/ws/telemetry?astronaut_id=<N>

    Each tick delivers a JSON object containing vitals, suit environment data,
    medical alert codes, and mission-elapsed time.
    Packets are simultaneously stored in the store-and-forward buffer.
    """
    await websocket.accept()
    state.active_websockets += 1
    client = websocket.client
    logger.info(
        "WebSocket connected — astronaut_id=%d  client=%s:%s  "
        "active_streams=%d",
        astronaut_id, client.host if client else "?", client.port if client else "?",
        state.active_websockets,
    )

    # Validate astronaut before streaming starts
    if state.df is not None and state.get_astronaut(astronaut_id) is None:
        await websocket.send_json({
            "error": f"Astronaut ID {astronaut_id} not found. Valid range: 1–75.",
            "code":  "INVALID_ASTRONAUT_ID",
        })
        await websocket.close(code=4004)
        return  # The finally block will naturally decrement state.active_websockets

    mission_start = time.monotonic()

    try:
        while True:
            tick_start = time.monotonic()
            elapsed    = tick_start - mission_start
            t          = time.time()

            packet = generate_telemetry_tick(
                astronaut_id    = astronaut_id,
                t               = t,
                elapsed_seconds = elapsed,
            )

            # Push to store-and-forward buffer (deep-space latency engine)
            state.store_forward_queue.append(packet)

            # Transmit to connected client
            await websocket.send_json(packet)

            # Log alerts to server console
            for alert in packet.get("alerts", []):
                logger.warning(
                    "ALERT [%s][%s] — %s | %s",
                    alert["severity"], alert["code"],
                    alert["message"],
                    ", ".join(alert.get("triggers", [])),
                )

            # Maintain 1 Hz tick rate (compensate for processing time)
            elapsed_this_tick = time.monotonic() - tick_start
            sleep_time = max(0.0, TELEMETRY_HZ - elapsed_this_tick)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected — astronaut_id=%d  active_streams=%d",
            astronaut_id, state.active_websockets - 1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "WebSocket error (astronaut_id=%d): %s", astronaut_id, exc, exc_info=True
        )
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass
    finally:
        state.active_websockets = max(0, state.active_websockets - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assert_csv_loaded() -> None:
    """Raise 503 if the astronaut CSV was not loaded successfully."""
    if state.df is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Astronaut baseline CSV not available "
                f"(expected at '{CSV_PATH}'). "
                "Place 'astronaut_vitals.csv' alongside server.py and restart."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host        = "0.0.0.0",
        port        = 8000,
        reload      = False,
        log_level   = "info",
        ws_ping_interval = 20,
        ws_ping_timeout  = 10,
    )
