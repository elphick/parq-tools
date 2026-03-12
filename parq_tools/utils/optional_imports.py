def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        def dummy(iterable, *args, **kwargs):
            return iterable
        return dummy


def get_ydata_profile_report(feature: str = "profiling"):
    """Lazily import and return ydata_profiling.ProfileReport.

    Raises a user-friendly ImportError if ydata_profiling is not installed.
    """
    try:
        from ydata_profiling import ProfileReport  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - message checked in tests
        raise ImportError(
            "ydata_profiling is required for profiling features in parq_tools. "
            f"Install it with 'pip install ydata-profiling' to use {feature}."
        ) from exc
    return ProfileReport
