from typing import Optional


def estimate_tokens(text: str) -> int:
    """Return a provider-agnostic estimate of the token count for ``text``."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def limit_text(
    text: str,
    max_tokens: Optional[int],
    marker: str = "\n\n[...truncated]",
) -> str:
    """Truncate ``text`` to roughly ``max_tokens`` tokens.

    The estimation is deliberately conservative and never splits by tokenizer,
    so callers should treat the limit as an upper bound rather than an exact
    token budget.
    """
    if not max_tokens or max_tokens <= 0:
        return text
    if estimate_tokens(text) <= max_tokens:
        return text

    max_chars = max(0, max_tokens * 4 - len(marker))
    return text[:max_chars].rstrip() + marker
