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


def verify_clerk_token(token: str) -> str:
    """
    Purpose: Verifies a Clerk session token's signature and claims, and
    returns the verified user id it belongs to. This is the only
    trustworthy source of "who is this" in the backend — a client-
    supplied id (like the old session_id) should never be used to
    identify a user once this exists.

    Args:
        token (str): The raw Clerk session token, as sent by the
            frontend (typically in an "Authorization: Bearer <token>"
            header).

    Returns:
        str: The verified Clerk user id (the token's "sub" claim).

    Raises:
        jwt.InvalidTokenError (or a subclass, e.g. ExpiredSignatureError,
            InvalidSignatureError): If the token is expired, malformed,
            signed by the wrong key, or otherwise invalid.
    """
    signing_key = _jwks_client.get_signing_key_from_jwt(token)
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=CLERK_ISSUER,
        options={"require": ["exp", "iat", "sub"]},
    )
    return payload["sub"]
