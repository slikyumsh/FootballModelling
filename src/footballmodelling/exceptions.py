"""Custom exceptions for the football modelling package."""


class FootballModellingError(Exception):
    """Base exception for all project-specific errors."""


class ConfigError(FootballModellingError):
    """Raised when configuration is missing or invalid."""


class AssetNotFoundError(FootballModellingError):
    """Raised when required assets (e.g., background image) are not found."""


class SimulationError(FootballModellingError):
    """Raised when simulation cannot continue due to invalid state."""
