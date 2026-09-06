import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import api

# .env is loaded once inside common.config (against REPO_ROOT), so importing
# these works regardless of the directory this script is run from.
from common import config

_MISSING = [
    name for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    if not getattr(config, name)
]
if _MISSING:
    raise SystemExit(
        "Missing required LiveKit credential(s): "
        + ", ".join(_MISSING)
        + f"\nSet them in {config.REPO_ROOT / '.env'} (see .env.example) or in the shell."
    )

URL    = config.LIVEKIT_URL
KEY    = config.LIVEKIT_API_KEY
SECRET = config.LIVEKIT_API_SECRET

def make_token(identity: str, room: str = "demo",
               publish: bool = True, subscribe: bool = True) -> str:
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room,
            can_publish=publish,
            can_subscribe=subscribe,
        ))
        .to_jwt()
    )

if __name__ == "__main__":
    who = sys.argv[1] if len(sys.argv) > 1 else "caller"
    # Print both, labelled, so they can be pasted straight into meet.livekit.io.
    print(f"URL:   {URL}")
    print(f"TOKEN: {make_token(who)}")