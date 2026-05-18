"""Quick auth check: prints Kite profile if API key + access token are valid."""
import os
import sys

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY = os.environ.get("KITE_API_KEY", "").strip()
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "").strip()

if not API_KEY or not ACCESS_TOKEN:
    print("Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env.", file=sys.stderr)
    sys.exit(1)

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)
print(kite.profile())
