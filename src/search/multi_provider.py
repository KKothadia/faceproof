import logging
from typing import Any, List, Optional

import numpy as np

from src.schemas import SearchCandidate
from src.exceptions import SearchError

logger = logging.getLogger(__name__)

class MultiProviderSearchClient:
    """
    Tries each configured search provider in order, falling through to the next one only
    if the current provider errors out or genuinely returns zero candidates. The first
    provider to return any candidates wins - later providers are never consulted, so a
    provider is never used to "pad out" or override another's real result.

    With a single provider configured this behaves exactly like calling that provider
    directly - the fallback logic is a no-op until a second provider is actually available.
    """

    def __init__(self, providers: List[Any]):
        if not providers:
            raise SearchError("MultiProviderSearchClient requires at least one provider.")
        self.providers = providers

    def search_local_image(self, image: np.ndarray, artifact_dir: Optional[str] = None) -> List[SearchCandidate]:
        last_error: Optional[SearchError] = None
        any_provider_completed = False

        for provider in self.providers:
            name = type(provider).__name__
            try:
                candidates = provider.search_local_image(image, artifact_dir=artifact_dir)
            except SearchError as e:
                logger.warning("Search provider %s failed (%s) - trying next provider if available", name, e)
                last_error = e
                continue

            any_provider_completed = True
            if candidates:
                return candidates

            logger.info("Search provider %s returned zero candidates - trying next provider if available", name)

        # If every provider errored out (none even completed a search), surface that error
        # rather than silently reporting "no candidates" - a rate limit or auth failure is a
        # different outcome from a provider that ran cleanly and genuinely found nothing.
        if not any_provider_completed and last_error:
            raise last_error

        return []
