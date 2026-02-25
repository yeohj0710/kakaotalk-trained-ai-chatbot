from __future__ import annotations

import os
import struct
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


MAGIC = b"KTBENC01"
VERSION = 1
TAG_SIZE = 16
DEFAULT_ITERATIONS = 390000
HEADER_STRUCT = struct.Struct(">B I H H")


def _derive_key(secret: str, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(secret.encode("utf-8"))


def encrypt_file(
    source_path: str | Path,
    target_path: str | Path,
    secret: str,
    iterations: int = DEFAULT_ITERATIONS,
    chunk_size: int = 4 * 1024 * 1024,
) -> None:
    src = Path(source_path)
    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive_key(secret=secret, salt=salt, iterations=iterations)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()

    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(MAGIC)
        fout.write(HEADER_STRUCT.pack(VERSION, iterations, len(salt), len(nonce)))
        fout.write(salt)
        fout.write(nonce)
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            fout.write(encryptor.update(chunk))
        fout.write(encryptor.finalize())
        fout.write(encryptor.tag)


def _read_header(fin) -> tuple[int, bytes, bytes]:
    magic = fin.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError("Invalid encrypted file header (magic mismatch).")
    version, iterations, salt_len, nonce_len = HEADER_STRUCT.unpack(fin.read(HEADER_STRUCT.size))
    if version != VERSION:
        raise ValueError(f"Unsupported encrypted file version: {version}")
    salt = fin.read(salt_len)
    nonce = fin.read(nonce_len)
    if len(salt) != salt_len or len(nonce) != nonce_len:
        raise ValueError("Corrupted encrypted file header.")
    return iterations, salt, nonce


def decrypt_file(
    source_path: str | Path,
    target_path: str | Path,
    secret: str,
    chunk_size: int = 4 * 1024 * 1024,
) -> None:
    src = Path(source_path)
    dst = Path(target_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("rb") as fin:
        iterations, salt, nonce = _read_header(fin)
        key = _derive_key(secret=secret, salt=salt, iterations=iterations)
        decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).decryptor()

        remaining = src.stat().st_size - (
            len(MAGIC) + HEADER_STRUCT.size + len(salt) + len(nonce)
        )
        if remaining <= TAG_SIZE:
            raise ValueError("Encrypted payload is too small.")
        ciphertext_len = remaining - TAG_SIZE

        with dst.open("wb") as fout:
            to_read = ciphertext_len
            while to_read > 0:
                chunk = fin.read(min(chunk_size, to_read))
                if not chunk:
                    raise ValueError("Unexpected EOF while reading encrypted payload.")
                to_read -= len(chunk)
                fout.write(decryptor.update(chunk))
            tag = fin.read(TAG_SIZE)
            if len(tag) != TAG_SIZE:
                raise ValueError("Missing authentication tag.")
            fout.write(decryptor.finalize_with_tag(tag))


def decrypt_file_to_bytes(
    source_path: str | Path,
    secret: str,
    chunk_size: int = 4 * 1024 * 1024,
) -> bytes:
    src = Path(source_path)
    with src.open("rb") as fin:
        iterations, salt, nonce = _read_header(fin)
        key = _derive_key(secret=secret, salt=salt, iterations=iterations)
        decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).decryptor()

        remaining = src.stat().st_size - (
            len(MAGIC) + HEADER_STRUCT.size + len(salt) + len(nonce)
        )
        if remaining <= TAG_SIZE:
            raise ValueError("Encrypted payload is too small.")
        ciphertext_len = remaining - TAG_SIZE

        out = bytearray()
        to_read = ciphertext_len
        while to_read > 0:
            chunk = fin.read(min(chunk_size, to_read))
            if not chunk:
                raise ValueError("Unexpected EOF while reading encrypted payload.")
            to_read -= len(chunk)
            out.extend(decryptor.update(chunk))
        tag = fin.read(TAG_SIZE)
        if len(tag) != TAG_SIZE:
            raise ValueError("Missing authentication tag.")
        out.extend(decryptor.finalize_with_tag(tag))
        return bytes(out)
