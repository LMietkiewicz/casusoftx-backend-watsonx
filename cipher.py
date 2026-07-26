"""cipher.py - AES-256-GCM encryption for config secrets.

The key is derived from a
passphrase via PBKDF2-HMAC-SHA256; each value gets a fresh random salt and IV,
stored as::

    ENC( base64( salt(16) || iv(12) || ciphertext+tag ) )

GCM is authenticated, so a wrong passphrase or a tampered value fails on decrypt
rather than returning garbage.

CLI::

    python cipher.py encrypt          # prompts for value + passphrase
    python cipher.py decrypt 'ENC(...)'
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PREFIX = "ENC("
SUFFIX = ")"
SALT_LEN = 16         # bytes
IV_LEN = 12           # bytes (96-bit nonce — the GCM standard)
TAG_BITS = 128        # GCM tag length
KEY_BITS = 256        # AES-256
ITERATIONS = 600_000  # PBKDF2 rounds 


class SecretCipherError(Exception):
    """Encryption or decryption failed."""


def is_encrypted(value: Optional[str]) -> bool:
    """True if the value carries the ENC(...) envelope."""
    return bool(value) and value.startswith(PREFIX) and value.endswith(SUFFIX)


def wrap(b64: str) -> str:
    return f"{PREFIX}{b64}{SUFFIX}"


def unwrap(enc: str) -> str:
    return enc[len(PREFIX):-len(SUFFIX)]


def _derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS, dklen=KEY_BITS // 8
    )


def encrypt(plaintext: str, password: str) -> str:
    """Encrypt a value and return it wrapped as ``ENC(base64)``."""
    try:
        salt = os.urandom(SALT_LEN)
        iv = os.urandom(IV_LEN)
        aesgcm = AESGCM(_derive_key(password, salt))
        ct = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
        return wrap(base64.b64encode(salt + iv + ct).decode("ascii"))
    except Exception as exc:
        raise SecretCipherError("Encryption failed") from exc


def decrypt(token: str, password: str) -> str:
    """Decrypt an ``ENC(...)`` token (or a bare base64 payload)."""
    try:
        raw = base64.b64decode(unwrap(token) if is_encrypted(token) else token)
        salt = raw[:SALT_LEN]
        iv = raw[SALT_LEN:SALT_LEN + IV_LEN]
        ct = raw[SALT_LEN + IV_LEN:]
        return AESGCM(_derive_key(password, salt)).decrypt(iv, ct, None).decode("utf-8")
    except Exception as exc:
        raise SecretCipherError(
            "Decryption failed (wrong passphrase or corrupted value?)"
        ) from exc


def resolve(value: Optional[str], password: Optional[str]) -> Optional[str]:
    """Return a config value, decrypting it if it is enveloped.

    Plain values pass through untouched, so a config file can mix encrypted and
    plaintext entries during migration.
    """
    if value is None or not is_encrypted(value):
        return value
    if not password:
        raise SecretCipherError(
            "Found an ENC(...) value but no passphrase is set "
            "(expected in CONFIG_PASSPHRASE)"
        )
    return decrypt(value, password)


if __name__ == "__main__":
    import argparse
    import getpass

    parser = argparse.ArgumentParser(description="Encrypt or decrypt a config secret.")
    parser.add_argument("mode", choices=["encrypt", "decrypt"])
    parser.add_argument("value", nargs="?", help="value (prompted if omitted)")
    args = parser.parse_args()

    val = args.value or getpass.getpass("Value: ")
    pw = os.getenv("CONFIG_PASSPHRASE") or getpass.getpass("Passphrase: ")
    print(encrypt(val, pw) if args.mode == "encrypt" else decrypt(val, pw))