import logging
import os
import string
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SOULS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "souls")
)


def normalize_role_name(role_name: str) -> str:
    """Normalizes role name by stripping suffixes, lowercase, and formatting cleanly."""
    name = role_name.lower().strip()
    if name.endswith(".md"):
        name = name[:-3]
    if name.endswith("_soul"):
        name = name[:-5]
    return name


class SoulLoader:
    """Loader class managing soul prompts with caching and formatting capabilities."""

    def __init__(self, souls_dir: Optional[str] = None):
        self.souls_dir = (
            souls_dir
            or os.getenv("SOULS_DIR")
            or DEFAULT_SOULS_DIR
        )
        self._cache: Dict[str, str] = {}

    def resolve_soul_path(self, role_name: str) -> str:
        """Resolves the absolute path for a given role name."""
        clean_role = normalize_role_name(role_name)
        filename = f"{clean_role}_soul.md"
        return os.path.join(self.souls_dir, filename)

    def load_soul(
        self,
        role_name: str,
        fallback_prompt: str = "",
        reload: bool = False,
    ) -> str:
        """Loads and caches persona rules for a role."""
        clean_role = normalize_role_name(role_name)

        if not reload and clean_role in self._cache:
            return self._cache[clean_role]

        file_path = self.resolve_soul_path(clean_role)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    self._cache[clean_role] = content
                    return content
            except Exception as exc:
                logger.warning(
                    "Failed to read soul file '%s': %s", file_path, exc, exc_info=True
                )
        else:
            logger.debug(
                "Soul file for role '%s' not found at '%s'. Using fallback.",
                role_name,
                file_path,
            )

        return fallback_prompt

    def format_soul(
        self,
        role_name: str,
        fallback_prompt: str = "",
        reload: bool = False,
        **kwargs: Any,
    ) -> str:
        """Loads persona rules and safely formats template variables."""
        content = self.load_soul(
            role_name, fallback_prompt=fallback_prompt, reload=reload
        )
        if not kwargs or not content:
            return content

        try:
            template = string.Template(content)
            return template.safe_substitute(**kwargs)
        except Exception as exc:
            logger.warning(
                "Failed to format soul template for '%s': %s", role_name, exc
            )
            return content

    def clear_cache(self, role_name: Optional[str] = None) -> None:
        """Clears specific cached role or the entire soul cache."""
        if role_name:
            clean_role = normalize_role_name(role_name)
            self._cache.pop(clean_role, None)
        else:
            self._cache.clear()

    def list_available_souls(self) -> List[str]:
        """Discovers all available soul file roles in the souls directory."""
        if not os.path.exists(self.souls_dir):
            return []

        souls = []
        try:
            for entry in os.listdir(self.souls_dir):
                if entry.endswith("_soul.md"):
                    role_name = entry[:-8]  # Strip '_soul.md'
                    souls.append(role_name)
        except Exception as exc:
            logger.warning(
                "Error listing souls directory '%s': %s", self.souls_dir, exc
            )

        return sorted(souls)


# Global default instance for easy module-level access
_default_loader = SoulLoader()


def load_soul(
    role_name: str,
    fallback_prompt: str = "",
    reload: bool = False,
    souls_dir: Optional[str] = None,
) -> str:
    """Module-level function loading persona rules for a role."""
    loader = _default_loader if souls_dir is None else SoulLoader(souls_dir=souls_dir)
    return loader.load_soul(role_name, fallback_prompt=fallback_prompt, reload=reload)


def format_soul(
    role_name: str,
    fallback_prompt: str = "",
    reload: bool = False,
    souls_dir: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Module-level function formatting persona templates for a role."""
    loader = _default_loader if souls_dir is None else SoulLoader(souls_dir=souls_dir)
    return loader.format_soul(
        role_name, fallback_prompt=fallback_prompt, reload=reload, **kwargs
    )


def clear_soul_cache(role_name: Optional[str] = None) -> None:
    """Module-level function clearing the soul loader cache."""
    _default_loader.clear_cache(role_name)


def list_available_souls(souls_dir: Optional[str] = None) -> List[str]:
    """Module-level function listing available soul roles."""
    loader = _default_loader if souls_dir is None else SoulLoader(souls_dir=souls_dir)
    return loader.list_available_souls()
