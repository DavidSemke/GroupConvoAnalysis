# expected format is 00:00.0
def convert_to_secs(timestamp):
    mins, secs = timestamp.split(':')
    return int(mins) * 60 + float(secs)


def convert_to_minutes(timestamp):
    mins, secs = timestamp.split(':')
    return int(mins) + float(secs)/60


def convert_to_timestamp(secs):
    secs = round(float(secs), 1)
    split = str(secs).split('.')
    secs, decisecs = [int(s) for s in split]
    mins = min(secs // 60, 99)
    leftover_secs = secs % 60
    return f"{mins:02d}:{leftover_secs:02d}.{decisecs}"