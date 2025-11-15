# models/backup.py
"""
Checkpoint backup utilities for Google Drive integration.

This module provides functions to backup model checkpoints to Google Drive,
useful for cloud-based training environments like Google Colab.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def setup_drive_backup(drive_backup_path: str) -> bool:
    """
    Setup Google Drive backup directory and check accessibility.

    Args:
        drive_backup_path: Path to Google Drive backup directory

    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Check if Google Drive is mounted (Colab environment)
        drive_path = Path(drive_backup_path)
        if not drive_path.parent.exists():
            print(
                "Google Drive not mounted or path not accessible. Skipping Drive backup."
            )
            return False

        # Create backup directories
        drive_path.mkdir(parents=True, exist_ok=True)
        (drive_path / "best_checkpoints").mkdir(exist_ok=True)
        (drive_path / "backup_history").mkdir(exist_ok=True)

        print(f"Google Drive backup setup at: {drive_path}")
        return True
    except Exception as e:
        print(f"Failed to setup Google Drive backup: {e}")
        return False


def backup_checkpoint_to_drive(
    checkpoint_path: Path, drive_backup_path: str
) -> bool:
    """
    Backup a checkpoint to Google Drive.

    Args:
        checkpoint_path: Local path to checkpoint file
        drive_backup_path: Google Drive directory path

    Returns:
        True if backup successful, False otherwise
    """
    try:
        if not checkpoint_path.exists():
            return False

        drive_path = Path(drive_backup_path)

        # Current backup location
        current_backup = drive_path / "best_checkpoints" / "best.pt"

        # Timestamped backup in history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_backup = drive_path / "backup_history" / f"best_{timestamp}.pt"

        # Copy to both locations
        shutil.copy2(checkpoint_path, current_backup)
        shutil.copy2(checkpoint_path, history_backup)

        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"Checkpoint backed up to Drive ({file_size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"Failed to backup to Drive: {e}")
        return False


def backup_final_checkpoint(
    final_checkpoint_path: Path, drive_backup_path: str
) -> bool:
    """
    Backup final training checkpoint to Google Drive history.

    Args:
        final_checkpoint_path: Local path to final checkpoint
        drive_backup_path: Google Drive directory path

    Returns:
        True if backup successful, False otherwise
    """
    try:
        drive_path = Path(drive_backup_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_backup = drive_path / "backup_history" / f"final_{timestamp}.pt"
        shutil.copy2(final_checkpoint_path, final_backup)
        print(f"Final checkpoint backed up: {final_backup}")
        return True
    except Exception as e:
        print(f"Failed to backup final checkpoint: {e}")
        return False
