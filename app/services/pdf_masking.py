from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
from werkzeug.datastructures.file_storage import FileStorage

from app import ImageMasker


class PDFMasker:
    """
    A class for masking PDF files by detecting areas containing faces and replacing them with black rectangles.
    """

    def __init__(self, image_masker: ImageMasker):
        """
        Initializes the PDFMasker with an ImageMasker instance.
        """
        self.image_masker = image_masker

    def mask_file(self, file: FileStorage, should_draw_gizmos: bool = False) -> bytes:
        """
        Masks the PDF file by detecting areas with faces and replacing them with black rectangles.
        :param file: The PDF data to mask.
        :param should_draw_gizmos: Whether to draw gizmos around the detected areas.
        """
        return self.mask_data(file.read(), should_draw_gizmos)

    def mask_data(self, pdf_as_bytes: bytes, should_draw_gizmos: bool = False) -> bytes:
        """
        Masks the PDF data by detecting areas with faces and replacing them with black rectangles.
        :param pdf_as_bytes: The PDF data to mask.
        :param should_draw_gizmos: Whether to draw gizmos around the detected areas.
        :return: The masked PDF data.
        """
        pdf_document = fitz.open(stream=pdf_as_bytes, filetype="pdf")
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            page_width = page.mediabox.width
            page_height = page.mediabox.height
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                image_obj = pdf_document.extract_image(xref)
                # Check if the image has a soft mask. This is usually the case for images with transparency
                # like PNGs and GIFs. The soft mask can also occur in images that have ocr text on top of the
                # image. Those images should be ignored.
                # todo: This is a workaround for behavior in different PDF creation tools and has to be monitored.
                if image_obj["smask"] != 0 and not (image_obj["ext"] == 'png' or image_obj["ext"] == 'gif'):
                    continue

                image_bytes = image_obj["image"]
                image = Image.open(BytesIO(image_bytes))
                overlapping = self.is_overlapping_full_page(image, page_width, page_height)
                result, amount_of_masks = self.image_masker.mask_data(
                    image_bytes,
                    allow_full_mask=not overlapping,
                    should_draw_gizmos=should_draw_gizmos
                )
                if amount_of_masks > 0:
                    # Only replace the image if any faces were detected. This keeps the original
                    # PDF content intact as much as possible.
                    page.replace_image(xref, pixmap=fitz.Pixmap(BytesIO(result)))

        modified_pdf_as_bytes = pdf_document.tobytes()
        pdf_document.close()
        return modified_pdf_as_bytes

    def is_overlapping_full_page(self, image: Image, width: int, height: int) -> bool:
        """
        Checks if the image overlaps more than 80% of the page.
        :param image: The image to check.
        :param width: The width of the page.
        :param height: The height of the page.
        :return: True if the image overlaps more than 80% of the page, False otherwise.
        """
        image_width, image_height = image.size
        return image_width > 0.8 * width and image_height > 0.8 * height
