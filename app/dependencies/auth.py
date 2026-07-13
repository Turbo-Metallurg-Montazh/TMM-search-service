import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    from app.config import JWT_PUBLIC_KEY
except ImportError:
    import os
    JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", "your-default-public-key-here")

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authorization credentials"
            )
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Bearer required"
            )
        token = credentials.credentials
        try:
            payload = jwt.decode(token, JWT_PUBLIC_KEY, algorithms=["RS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token signature or structure"
            )

jwt_bearer = JWTBearer()

class VerifyScopes:
    def __init__(self, required_scopes: list[str]):
        self.required_scopes = required_scopes

    def __call__(self, payload: dict = Depends(jwt_bearer)):
        token_scopes = payload.get("scopes", [])
        for scope in self.required_scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient permissions (missing required scopes)"
                )
        return payload