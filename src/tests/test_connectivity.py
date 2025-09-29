import os
import socket
from urllib.parse import urlparse

import psycopg2
import pytest
from dotenv import load_dotenv

# Ensure .env is loaded so pytest sees DB credentials
load_dotenv()


def _redact_password_in_url(url: str) -> str:
    if not url:
        return url
    try:
        if "://" in url and "@" in url:
            start = url.find("://") + 3
            end = url.rfind("@")
            creds = url[start:end]
            if ":" in creds:
                user = creds.split(":", 1)[0]
                return url[:start] + f"{user}:****" + url[end:]
        return url
    except Exception:
        return url


def _print_dns_resolution(host: str, port: str) -> None:
    try:
        infos = socket.getaddrinfo(host, int(port), proto=socket.IPPROTO_TCP)
        ips = sorted({ai[4][0] for ai in infos})
        print(f"DNS {host}:{port} -> {ips}")
    except Exception as e:
        print(f"DNS resolution error for {host}:{port}: {e}")


def test_supabase_postgres_connect_env():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if "sslmode=" not in database_url:
            database_url = database_url + ("&sslmode=require" if "?" in database_url else "?sslmode=require")
        redacted = _redact_password_in_url(database_url)
        print(f"DATABASE_URL (redacted): {redacted}")
        try:
            parsed = urlparse(database_url)
            host = parsed.hostname or ""
            port = str(parsed.port or "")
            user = parsed.username or ""
            dbname = (parsed.path or "/").lstrip("/")
            print(f"Parsed -> user={user}, host={host}, port={port}, dbname={dbname}, sslmode=require")
            if host and port:
                _print_dns_resolution(host, port)
        except Exception as e:
            print(f"URL parse error: {e}")
        try:
            with psycopg2.connect(database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("select 1")
                    row = cur.fetchone()
                    assert row and row[0] == 1
        except psycopg2.OperationalError as e:
            print(f"psycopg2 OperationalError: {e}")
            raise
        return

    user = os.getenv("user")
    password = os.getenv("password")
    host = os.getenv("host")
    port = os.getenv("port")
    dbname = os.getenv("dbname")

    assert user and password and host and dbname, "DB env vars missing (user, password, host, dbname)"

    print(f"ENV -> user={user}, host={host}, port={port}, dbname={dbname}, sslmode=require")
    if host and port:
        _print_dns_resolution(host, port)

    try:
        with psycopg2.connect(user=user, password=password, host=host, port=port, dbname=dbname, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute("select 1")
                row = cur.fetchone()
                assert row and row[0] == 1
    except psycopg2.OperationalError as e:
        print(f"psycopg2 OperationalError: {e}")
        raise


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_openai_api_key_present():
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set"


