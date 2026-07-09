import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Псевдокод для получения публичного ключа из конфигурации
# В реальном коде импортируйте его из вашего app.config
try:
    from app.config import JWT_PUBLIC_KEY
except ImportError:
    # Заглушка, если переменная еще не добавлена в config.py
    import os

    JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY", "your-default-public-key-here")


# Наследуемся от нативного HTTPBearer, чтобы FastAPI автоматически
# добавил схему безопасности в OpenAPI/Swagger документацию
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
            # Декодируем и валидируем токен публичным ключом
            # Библиотека PyJWT сама проверяет 'exp' (время жизни) токена
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


# Фабрика зависимостей для проверки конкретных прав (Scopes)
class VerifyScopes:
    def __init__(self, required_scopes: list[str]):
        self.required_scopes = required_scopes

    def __call__(self, payload: dict = Depends(JWTBearer())):
        # Извлекаем scopes, которые Бэкенд зашил в payload токена
        token_scopes = payload.get("scopes", [])

        # Проверяем, что у токена есть все необходимые права для этого эндпоинта
        for scope in self.required_scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Insufficient permissions (missing required scopes)"
                )
        return payload