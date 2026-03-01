"""Run and training status types used across the stack."""

from typing import Literal

# Status as stored in the database
DBRunStatus = Literal["running", "completed", "failed", "stopped"]

# Status as broadcast over WebSocket / shown in the dashboard
WireStatus = Literal["training", "idle", "complete", "paused"]

# Mapping from DB status to wire status
DB_TO_WIRE: dict[str, WireStatus] = {
    "running": "training",
    "completed": "complete",
    "stopped": "idle",
    "failed": "idle",
}


def db_to_wire(db_status: str) -> WireStatus:
    """Convert a DB run status to the wire/dashboard status."""
    return DB_TO_WIRE.get(db_status, "idle")
