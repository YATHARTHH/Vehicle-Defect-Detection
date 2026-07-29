import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "inspections.db")
logger = logging.getLogger("overbody_api.database")


def init_db() -> None:
    """Initializes SQLite inspection database tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            image_hash TEXT,
            filename TEXT,
            timestamp TEXT,
            total_defects INTEGER,
            overall_severity TEXT,
            defects_json TEXT,
            repair_guide TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info(f"[database] SQLite database initialized successfully at {os.path.basename(DB_PATH)}")


def save_inspection(
    inspection_id: str,
    image_hash: str,
    filename: str,
    total_defects: int,
    overall_severity: str,
    defects: List[Dict[str, Any]],
    repair_guide: Optional[str] = None,
) -> None:
    """Saves a completed inspection audit record to SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO inspections 
            (id, image_hash, filename, timestamp, total_defects, overall_severity, defects_json, repair_guide)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                image_hash,
                filename,
                datetime.utcnow().isoformat() + "Z",
                total_defects,
                overall_severity,
                json.dumps(defects),
                repair_guide or "",
            ),
        )
        conn.commit()
        conn.close()
        logger.info(f"[database] Saved inspection audit record {inspection_id[:8]} to SQLite.")
    except Exception as e:
        logger.error(f"[database] Error saving inspection record to SQLite: {e}")


def get_recent_inspections(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves recent inspection audit records."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, image_hash, filename, timestamp, total_defects, overall_severity, defects_json
            FROM inspections
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append(
                {
                    "id": row[0],
                    "image_hash": row[1],
                    "filename": row[2],
                    "timestamp": row[3],
                    "total_defects": row[4],
                    "overall_severity": row[5],
                    "defects": json.loads(row[6]) if row[6] else [],
                }
            )
        return records
    except Exception as e:
        logger.error(f"[database] Error querying inspections: {e}")
        return []


def get_inspection_stats() -> Dict[str, Any]:
    """Computes inspection audit statistics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(total_defects), AVG(total_defects) FROM inspections")
        total_inspections, total_defects, avg_defects = cursor.fetchone()
        
        cursor.execute("SELECT overall_severity, COUNT(*) FROM inspections GROUP BY overall_severity")
        severity_counts = dict(cursor.fetchall())
        conn.close()

        return {
            "total_inspections": total_inspections or 0,
            "total_defects_found": total_defects or 0,
            "average_defects_per_vehicle": round(avg_defects or 0.0, 2),
            "severity_breakdown": severity_counts,
        }
    except Exception as e:
        logger.error(f"[database] Error computing stats: {e}")
        return {
            "total_inspections": 0,
            "total_defects_found": 0,
            "average_defects_per_vehicle": 0.0,
            "severity_breakdown": {},
        }
