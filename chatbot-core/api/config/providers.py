"""Load and validate the configured LLM provider catalog."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CATALOG_PATH = _CONFIG_DIR / "providers.json"
_PROVIDER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class ProviderDefinition(BaseModel):
    """Configured LLM provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    model: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Normalize and validate the provider ID used for routing."""
        value = value.strip().lower()
        if not _PROVIDER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with a lowercase letter and contain only "
                "lowercase letters, digits, and underscores"
            )
        return value

    @field_validator("label", "model")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Trim and reject blank provider metadata."""
        value = value.strip()
        if not value:
            raise ValueError("cannot be blank")
        return value

    @property
    def api_key_env(self) -> str:
        """Return the environment variable used for this provider's API key."""
        return f"{self.id.upper()}_API_KEY"


class ProviderCatalog(BaseModel):
    """Configured provider catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: tuple[ProviderDefinition, ...] = Field(min_length=1)


def load_provider_catalog(
    catalog_path: Path = _DEFAULT_CATALOG_PATH,
) -> tuple[ProviderDefinition, ...]:
    """Load and validate providers.json."""
    try:
        with catalog_path.open("r", encoding="utf-8") as catalog_file:
            data = json.load(catalog_file)

        providers = ProviderCatalog.model_validate(data).providers

    except FileNotFoundError as error:
        raise ValueError(f"Provider catalog not found: {catalog_path}") from error

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in provider catalog: line {error.lineno}, "
            f"column {error.colno}"
        ) from error

    except ValidationError as error:
        raise ValueError(f"Invalid provider configuration:\n{error}") from error

    provider_ids = [provider.id for provider in providers]
    if len(provider_ids) != len(set(provider_ids)):
        raise ValueError("Provider IDs must be unique.")

    return providers
