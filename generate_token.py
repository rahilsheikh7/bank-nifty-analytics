"""
One-time (or periodic) login flow: opens login URL, exchanges request_token for access_token.
Credentials are read from environment / .env — do not hard-code secrets.
"""
import os
import sys

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY = os.environ.get("KITE_API_KEY", "").strip()
API_SECRET = os.environ.get("KITE_API_SECRET", "").strip()

if not API_KEY or not API_SECRET:
    print(
        "Missing KITE_API_KEY or KITE_API_SECRET.\n"
        "Copy .env.example to .env and set both variables.",
        file=sys.stderr,
    )
    sys.exit(1)

kite = KiteConnect(api_key=API_KEY)

print("Open this URL in your browser:")
print(kite.login_url())
print()

request_token = input("Enter request_token from the redirect URL: ").strip()
if not request_token:
    print("No request_token provided.", file=sys.stderr)
    sys.exit(1)

data = kite.generate_session(request_token=request_token, api_secret=API_SECRET)
access_token = data["access_token"]

print("\nAccess Token:")
print(data)
print(
    "\nAdd this line to your .env file:\n"
    f'KITE_ACCESS_TOKEN="{access_token}"'
)
