from pathlib import Path


PROPOSAL_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PROPOSAL_DIR / "proposal.md"

SECTION_FILES = [
    "00_Introduction.md",
    "01_Background.md",
    "02_Product_Ecommerce.md",
    "03_Datasets.md",
    "04_Problem_Statement.md",
    "05_Implementation_Challenges.md",
    "06_Related_Works.md",
    "07_Methodology.md",
    "08_Improvements.md",
    "09_Expected_Result.md",
    "10_Appendix_and_Reference.md",
]


def merge_markdown_files() -> None:
    parts: list[str] = []

    for filename in SECTION_FILES:
        path = PROPOSAL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing proposal section: {path}")

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Proposal section is empty: {path}")

        parts.append(content)

    OUTPUT_FILE.write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    merge_markdown_files()
    print(f"Merged {len(SECTION_FILES)} files into {OUTPUT_FILE}")
