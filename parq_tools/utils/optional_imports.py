def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        def dummy(iterable, *args, **kwargs):
            return iterable
        return dummy


def get_data_profile_report(feature: str = "profiling"):
    """Lazily import and return data_profiling.ProfileReport.

    Raises a user-friendly ImportError if data_profiling is not installed.
    """
    try:
        from data_profiling import ProfileReport  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - message checked in tests
        raise ImportError(
            "data_profiling is required for profiling features in parq_tools. "
            f"Install it with 'pip install fg-data-profiling' to use {feature}."
        ) from exc
    return ProfileReport


def get_data_profile_compare(feature: str = "profile comparison"):
    """Lazily import and return data_profiling.compare_reports.compare."""
    try:
        from data_profiling.compare_reports import compare  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - message checked in tests
        raise ImportError(
            "data_profiling is required for profiling features in parq_tools. "
            f"Install it with 'pip install fg-data-profiling' to use {feature}."
        ) from exc
    return compare
