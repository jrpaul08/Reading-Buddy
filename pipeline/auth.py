"""
Verifies Clerk-issued auth tokens for the Reading Buddy backend.

Checks a token's signature directly against Clerk's published public
keys (via PyJWT's JWKS client, which fetches and caches them) rather
than trusting anything the frontend claims — this is what turns "the
frontend says this is user X" into "Clerk actually signed a token
proving this is user X".
"""

import jwt

CLERK_ISSUER = "https://communal-mammal-6980.clerk.accounts.dev"
CLERK_JWKS_URL = f"{CLERK_ISSUER}/.well-known/jwks.json"

# Module-level so it's created once per container and its internal key
# cache is reused across every request that container handles, instead
# of re-fetching Clerk's public keys every time.
_jwks_client = jwt.PyJWKClient(CLERK_JWKS_URL)


class AuthError(Exception):
    """Raised whenever a token can't be verified, for any reason —
    missing, malformed, expired, or signed by the wrong key. Callers
    only need to catch this one type, not know about PyJWT's internals."""


def verify_clerk_token(token: str) -> str:
    """
    Purpose: Verifies a Clerk session token's signature and claims, and
    returns the verified user id it belongs to. This is the only
    trustworthy source of "who is this logged-in user is" in the
    backend — a client-supplied id like a guest session_id is never
    verified this way, and is only ever trusted for guest mode, where
    there's no identity claim being made in the first place (see
    orchestrator_modal._resolve_identity).

    Args:
        token (str): The raw Clerk session token, as sent by the
            frontend (typically in an "Authorization: Bearer <token>"
            header).

    Returns:
        str: The verified Clerk user id (the token's "sub" claim).

    Raises:
        AuthError: If the token is expired, malformed, signed by the
            wrong key, or otherwise invalid.
    """
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.InvalidTokenError as e:
        raise AuthError(str(e)) from e

    return payload["sub"]
