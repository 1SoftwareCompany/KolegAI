from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import HTTPException, Security, logger, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import jwt

from jwt import PyJWKClient
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidTokenError,
    PyJWKClientError,
)

bearer_scheme = HTTPBearer(auto_error=False)


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": "Bearer"},
    )


@dataclass(frozen=True)
class JwtConfig:
    issuer: str
    audience: str
    jwks_uri: str
    alg: str = "RS256"


class JwtVerifier:

    def __init__(self, cfg: JwtConfig):
        self.cfg = cfg
        self.jwks_client = PyJWKClient(cfg.jwks_uri)
        
    def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    ) -> Dict[str, Any]:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise unauthorized()

        token = credentials.credentials

        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token).key

            claims = jwt.decode(
                token,
                signing_key,
                algorithms=[self.cfg.alg],
                issuer=self.cfg.issuer,
                audience=self.cfg.audience,
                options={"require": ["exp", "iss", "aud"]}
            )
            return claims

        except ExpiredSignatureError:
            logger.logger.error("Invalid signature")
            raise unauthorized()
        except InvalidAudienceError:
            logger.logger.error("Invalid audience")
            raise unauthorized()
        except InvalidIssuerError:
            logger.logger.error("Invalid issuer")
            raise unauthorized()
        except (PyJWKClientError, InvalidTokenError):
            logger.logger.error("Invalid token")
            raise unauthorized()
        except Exception:
            logger.logger.error("Invalid token")
            raise unauthorized()
