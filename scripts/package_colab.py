"""Create the small source bundle uploaded by the Stage 3 Colab notebook."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "openinfer_stage3_bundle.zip"
SOURCES = [ROOT / "requirements.txt", *sorted((ROOT / "gateway").rglob("*.py"))]

with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as bundle:
    for source in SOURCES:
        bundle.write(source, source.relative_to(ROOT))

print(f"Created {OUTPUT} with {len(SOURCES)} files")
