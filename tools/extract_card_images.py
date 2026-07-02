"""Extract one JPEG per card from docs/Card_ID List_JP.pdf."""

from io import BytesIO
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "docs" / "Card_ID List_JP.pdf"
OUTPUT_DIR = ROOT / "docs" / "cards"
FIRST_CARD_PAGE = 40  # One-based; pages 1-39 contain the card ID table.


def main() -> None:
    reader = PdfReader(PDF_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_count = len(reader.pages) - FIRST_CARD_PAGE + 1
    written = 0

    for card_id, page in enumerate(reader.pages[FIRST_CARD_PAGE - 1 :], start=1):
        xobjects = page["/Resources"]["/XObject"].get_object()
        images = [obj.get_object() for obj in xobjects.values() if obj.get_object().get("/Subtype") == "/Image"]
        if len(images) != 1:
            raise RuntimeError(f"Card ID {card_id}: expected one image, found {len(images)}")

        image = images[0]
        card = Image.open(BytesIO(image.get_data())).convert("RGB")
        if card.size != (600, 825):
            raise RuntimeError(f"Card ID {card_id}: unexpected size {card.width}x{card.height}")

        # The PDF stores rounded corners and some card details in a soft mask.
        # Flatten the image and its soft mask onto white so ordinary JPEG viewers
        # reproduce the same appearance as the PDF page.
        flattened = Image.new("RGB", card.size, "white")
        smask = image.get("/SMask")
        if smask is None:
            flattened.paste(card)
        else:
            alpha = Image.open(BytesIO(smask.get_object().get_data())).convert("L")
            if alpha.size != card.size:
                raise RuntimeError(f"Card ID {card_id}: soft mask size does not match image")
            flattened.paste(card, mask=alpha)

        flattened.save(
            OUTPUT_DIR / f"{card_id:04d}.jpg",
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        written += 1

    if written != expected_count:
        raise RuntimeError(f"Expected {expected_count} cards, wrote {written}")

    print(f"Extracted {written} cards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
