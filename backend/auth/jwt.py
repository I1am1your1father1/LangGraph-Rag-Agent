from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from backend.config import JWT_ALGORITHM, JWT_SECRET_KEY


def create_access_token(
    user_id: str,
    expires_minutes: int = 60,
) -> str:
    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )
        return payload

    except jwt.ExpiredSignatureError:
        raise ValueError("token 已过期")

    except jwt.InvalidTokenError:
        raise ValueError("无效 token")