"""
Authentication middleware to populate context state with user information
"""
import jwt
import logging
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers, get_http_request

from auth.oauth21_session_store import ensure_session_from_access_token
from auth.oauth_config import get_oauth_config
from google.oauth2.credentials import Credentials as GoogleCredentials
from google.auth.transport.requests import Request as GoogleAuthRequest

# Configure logging
logger = logging.getLogger(__name__)


class AuthInfoMiddleware(Middleware):
    """
    Middleware to extract authentication information from JWT tokens
    and populate the FastMCP context state for use in tools and prompts.
    """
    
    def __init__(self):
        super().__init__()
        self.auth_provider_type = "GoogleProvider"
    
    async def _process_request_for_auth(self, context: MiddlewareContext):
        """Helper to extract, verify, and store auth info from a request."""
        if not context.fastmcp_context:
            logger.warning("No fastmcp_context available")
            return

        logger.info(f"Processing request for auth: {context.fastmcp_context}")
        req = get_http_request()
        if req is not None:
            body_bytes = await req.body()
            logger.info(f"{body_bytes=}")
        headers = get_http_headers()
        if headers:
            try:
                refresh_token = headers.get("x-refresh-token")
                logger.info(f"{refresh_token is not None=}")
                cfg = get_oauth_config()
                creds = GoogleCredentials(
                    token=None,
                    refresh_token=refresh_token,
                    client_id=cfg.client_id,
                    client_secret=cfg.client_secret,
                    token_uri="https://oauth2.googleapis.com/token",
                )
                creds.refresh(GoogleAuthRequest())
                context.fastmcp_context.set_state("credentials", creds)
                context.fastmcp_context.set_state("credentials", creds)
                return
            except Exception as e:
                logger.error(f"Failed refresh-token header auth: {e}")
    
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Extract auth info from token and set in context state"""
        logger.debug("Processing tool call authentication")
        
        try:
            await self._process_request_for_auth(context)
            
            logger.debug("Passing to next handler")
            result = await call_next(context)
            logger.debug("Handler completed")
            return result
            
        except Exception as e:
            # Check if this is an authentication error - don't log traceback for these
            if "GoogleAuthenticationError" in str(type(e)) or "Access denied: Cannot retrieve credentials" in str(e):
                logger.info(f"Authentication check failed: {e}")
            else:
                logger.error(f"Error in on_call_tool middleware: {e}", exc_info=True)
            raise
    
    async def on_get_prompt(self, context: MiddlewareContext, call_next):
        """Extract auth info for prompt requests too"""
        logger.debug("Processing prompt authentication")
        
        try:
            await self._process_request_for_auth(context)
            
            logger.debug("Passing prompt to next handler")
            result = await call_next(context)
            logger.debug("Prompt handler completed")
            return result
            
        except Exception as e:
            # Check if this is an authentication error - don't log traceback for these
            if "GoogleAuthenticationError" in str(type(e)) or "Access denied: Cannot retrieve credentials" in str(e):
                logger.info(f"Authentication check failed in prompt: {e}")
            else:
                logger.error(f"Error in on_get_prompt middleware: {e}", exc_info=True)
            raise
