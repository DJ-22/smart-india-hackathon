import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()
KEY    = os.environ["LIVEKIT_API_KEY"]
SECRET = os.environ["LIVEKIT_API_SECRET"]
URL    = os.environ["LIVEKIT_URL"]

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
    import sys
    who = sys.argv[1] if len(sys.argv) > 1 else "caller"
    print(make_token(who))