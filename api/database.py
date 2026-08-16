"""
Persistence for analyses, radiologist feedback, and patient timelines.

SQLite by default; point DATABASE_URL at Postgres for a real deployment.
Storing results is what makes longitudinal tracking, audit trails, and
hard-example mining possible — none of which work with stateless inference.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

Base = declarative_base()

DEFAULT_DB_URL = "sqlite:///./brainscan.db"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class Study(Base):
    """One analysed MRI scan."""

    __tablename__ = "studies"

    id = Column(String(32), primary_key=True, default=_new_id)
    created_at = Column(DateTime, default=_utcnow, nullable=False, index=True)

    patient_id = Column(String(64), index=True, nullable=True)
    filename = Column(String(255), nullable=True)
    image_sha256 = Column(String(64), index=True, nullable=False)
    study_date = Column(DateTime, nullable=True)

    predicted_class = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=True)
    probabilities = Column(Text, nullable=True)  # JSON blob

    tumor_volume_mm3 = Column(Float, nullable=True)
    tumor_area_px = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    sequence_type = Column(String(16), nullable=True)
    who_grade = Column(String(16), nullable=True)

    ood_score = Column(Float, nullable=True)
    is_ood = Column(Boolean, default=False, nullable=False)

    flagged_for_review = Column(Boolean, default=False, nullable=False)
    model_version = Column(String(64), nullable=False)
    git_sha = Column(String(40), nullable=True)

    gradcam_overlay = Column(Text, nullable=True)  # base64 PNG, for report rendering

    feedback = relationship(
        "Feedback", back_populates="study", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (Index("ix_studies_patient_date", "patient_id", "created_at"),)


class Feedback(Base):
    """Radiologist verdict on a study — the retraining signal."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String(32), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False, unique=True)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    reviewer = Column(String(128), nullable=True)
    corrected_class = Column(String(32), nullable=False)
    agrees_with_ai = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)

    study = relationship("Study", back_populates="feedback")


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, future=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def reset_engine() -> None:
    """Drop cached engine so a new DATABASE_URL takes effect — used by tests."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


def image_fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
