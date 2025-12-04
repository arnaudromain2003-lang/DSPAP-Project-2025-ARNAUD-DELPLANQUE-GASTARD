def intervals_to_duration(interval_count, interval_minutes=15):
    """
    Convert a number of fixed-length time intervals (default: 15 minutes)
    into a human-readable duration expressed in days, hours, and minutes.

    Example:
        4 intervals  ->  1 hour
        96 intervals ->  1 day

    Args:
        interval_count (int): Number of intervals.
        interval_minutes (int): Length of one interval in minutes.

    Returns:
        str: Readable duration (e.g. '3 days 5 hours 45 minutes').
    """
    total_minutes = interval_count / interval_minutes

    hours, minutes = divmod(total_minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days:
        parts.append(f"{days} days")
    if hours:
        parts.append(f"{hours} hours")
    if minutes:
        parts.append(f"{int(minutes)} minutes")

    return " ".join(parts) if parts else "0 minutes"
