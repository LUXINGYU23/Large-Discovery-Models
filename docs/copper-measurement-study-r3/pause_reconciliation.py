from pathlib import Path
import shutil

def archive_pauses(queue, archive, expected):
    queue, archive = Path(queue), Path(archive)
    actual = {str(p.relative_to(queue)): p.read_bytes() for p in queue.rglob("PAUSED.json")}
    assert actual == expected, "pause evidence changed; reconciliation required"
    assert not archive.exists(), "pause archive already exists"
    for name in sorted(expected, key=lambda x: x == "PAUSED.json"):
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(queue / name, target)
    assert not list(queue.rglob("PAUSED.json")), "unreconciled scheduler pause"
