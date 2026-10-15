"""Server-side secret storage.

Secrets live in the macOS Keychain (service "recruiter-voice-agent"), or in environment
variables, which take precedence. Values are never printed or logged.

Usage:
    python -m backend.keystore set OPENAI_API_KEY   # native dialog, hidden input
    python -m backend.keystore check                # shows which secrets are present
"""

from __future__ import annotations

import os
import subprocess
import sys

import keyring

SERVICE = "recruiter-voice-agent"
KNOWN = ("OPENAI_API_KEY", "GEMINI_API_KEY", "AZURE_SPEECH_KEY", "ANTHROPIC_API_KEY")


def get_secret(name: str) -> str:
    value = os.environ.get(name) or keyring.get_password(SERVICE, name)
    if not value:
        raise KeyError(f"{name} is not set (env or Keychain service '{SERVICE}')")
    return value


def has_secret(name: str) -> bool:
    return bool(os.environ.get(name) or keyring.get_password(SERVICE, name))


def store_secret(name: str, value: str) -> None:
    keyring.set_password(SERVICE, name, value)


def _prompt_hidden(name: str) -> str | None:
    script = (
        f'display dialog "Paste {name} for Recruiter Voice Agent" '
        'default answer "" with hidden answer with title "Recruiter Voice Agent" '
        'buttons {"Cancel", "Save"} default button "Save"\n'
        "text returned of result"
    )
    proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "set":
        name = argv[1]
        value = _prompt_hidden(name)
        if not value:
            print(f"{name}: cancelled or empty, nothing stored")
            return 1
        store_secret(name, value)
        print(f"{name}: stored in Keychain ({len(value)} chars)")
        return 0
    if argv and argv[0] == "check":
        for name in KNOWN:
            print(f"{name}: {'present' if has_secret(name) else 'missing'}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
