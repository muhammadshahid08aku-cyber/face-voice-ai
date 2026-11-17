"""Storage: SQLite locally, Supabase PostgreSQL in production. Templates are encrypted at rest (Fernet).

Only numeric embeddings are stored. Photos and audio are never written to disk or the database.
"""
import datetime as dt
import json
import os
import secrets
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from sqlalchemy import (Column, DateTime, Float, Integer, LargeBinary, MetaData, String, Table, Text, create_engine,
                        delete, func, insert, select, text, update)

md = MetaData()
persons = Table("persons", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ext_id", String(16), unique=True), Column("name", String(120)),
    Column("status", String(10)), Column("consent_version", String(20)), Column("consent_at", DateTime),
    Column("retention_until", DateTime), Column("created_at", DateTime), Column("erased_at", DateTime))
templates = Table("templates", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("person_id", Integer), Column("modality", String(10)), Column("blob", LargeBinary),
    Column("created_at", DateTime))
audit = Table("audit_log", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime), Column("actor", String(60)), Column("session", String(40)),
    Column("action", String(40)), Column("person_id", Integer), Column("decision", String(10)),
    Column("face_score", Float), Column("voice_score", Float), Column("detail", Text))


def now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def clean_url(url: str) -> str:
    """Tolerate common paste mistakes in a database URL (quotes, [brackets], unescaped password characters)."""
    url = url.strip().strip("\"'").replace("[", "").replace("]", "")
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" in rest:
        creds, host = rest.rsplit("@", 1)
        user, _, pw = creds.partition(":")
        rest = quote(unquote(user), safe=".") + ":" + quote(unquote(pw), safe="") + "@" + host
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+psycopg2"
    return scheme + "://" + rest


def load_key(database_url: str | None, key: str | None) -> bytes:
    if key:
        return key.encode()
    if database_url and not database_url.startswith("sqlite"):
        raise RuntimeError("BIOMETRIC_KEY is required with a remote database. Generate one with: "
                           "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    p = Path(__file__).parent / ".biometric_key"      # local development only
    if not p.exists():
        p.write_bytes(Fernet.generate_key())
        os.chmod(p, 0o600)
    return p.read_bytes().strip()


class Store:
    def __init__(self, url: str, key: bytes):
        self.engine = create_engine(clean_url(url), pool_pre_ping=True)
        self.fernet = Fernet(key)
        md.create_all(self.engine)

    # ---- people and templates
    def add_person(self, name, consent_version, retention_until):
        ext = "U-" + secrets.token_hex(3).upper()
        with self.engine.begin() as c:
            pid = c.execute(insert(persons).values(
                ext_id=ext, name=name.strip(), status="active", consent_version=consent_version,
                consent_at=now(), retention_until=retention_until, created_at=now())).inserted_primary_key[0]
        return pid, ext

    def add_template(self, person_id, modality, emb: np.ndarray):
        blob = self.fernet.encrypt(np.asarray(emb, dtype=np.float32).tobytes())
        with self.engine.begin() as c:
            c.execute(insert(templates).values(person_id=person_id, modality=modality, blob=blob, created_at=now()))

    def templates(self, modality: str) -> dict:
        with self.engine.connect() as c:
            rows = c.execute(select(templates.c.person_id, templates.c.blob)
                             .join(persons, persons.c.id == templates.c.person_id)
                             .where(templates.c.modality == modality, persons.c.status == "active")).all()
        out = {}
        for pid, blob in rows:
            out.setdefault(pid, []).append(np.frombuffer(self.fernet.decrypt(bytes(blob)), dtype=np.float32))
        return out

    def persons(self, include_erased=False) -> list[dict]:
        q = select(persons).order_by(persons.c.id)
        if not include_erased:
            q = q.where(persons.c.status == "active")
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(q)]

    def person(self, person_id=None, ext_id=None):
        q = select(persons)
        q = q.where(persons.c.id == person_id) if person_id is not None else q.where(persons.c.ext_id == ext_id)
        with self.engine.connect() as c:
            r = c.execute(q).first()
        return dict(r._mapping) if r else None

    def erase(self, person_id: int):
        with self.engine.begin() as c:
            c.execute(delete(templates).where(templates.c.person_id == person_id))
            c.execute(update(persons).where(persons.c.id == person_id)
                      .values(name="[erased]", status="erased", erased_at=now()))

    def purge_expired(self) -> int:
        with self.engine.connect() as c:
            ids = [r[0] for r in c.execute(select(persons.c.id).where(
                persons.c.status == "active", persons.c.retention_until < now()))]
        for i in ids:
            self.erase(i)
        return len(ids)

    # ---- audit
    def log(self, actor, action, person_id=None, decision=None, face=None, voice=None, detail="", session=None):
        with self.engine.begin() as c:
            c.execute(insert(audit).values(ts=now(), actor=actor, session=session, action=action, person_id=person_id,
                                           decision=decision, face_score=face, voice_score=voice,
                                           detail=detail if isinstance(detail, str) else json.dumps(detail)))

    def failures(self, person_id=None, session=None, minutes=10) -> int:
        since = now() - dt.timedelta(minutes=minutes)
        q = select(func.count()).select_from(audit).where(audit.c.ts >= since, audit.c.decision.in_(("REJECT", "REVIEW")))
        q = q.where(audit.c.person_id == person_id) if person_id is not None else q.where(audit.c.session == session)
        with self.engine.connect() as c:
            return int(c.execute(q).scalar() or 0)

    def audit_df(self, limit=500) -> pd.DataFrame:
        with self.engine.connect() as c:
            return pd.read_sql(text("select * from audit_log order by id desc limit :n"), c, params={"n": limit})
