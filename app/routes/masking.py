import io

from flask import Blueprint, jsonify, request, send_file

from app import cascade_classifiers, detection_model, image_processor
from app.services.image_masking import ImageMasker
from app.services.pdf_masking import PDFMasker

masking_blueprint = Blueprint('api', __name__, url_prefix='/api/v1/mask')

ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif'}


@masking_blueprint.route('/pdf', methods=['POST'])
def mask_pdf():
    """
    Endpoint for uploading a PDF file to be masked.
    The PDF is masked by detecting faces in all images inside the PDF and replacing
    them with black rectangles.
    Supported file type is PDF.
    :return: The masked PDF file.
    :raises: HTTP 400 if no file is uploaded.
    :raises: HTTP 415 if the file type is not supported.
    """
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"success": False, "message": "No file uploaded.", "code": 400}), 400

    file = request.files['file']

    if file.content_type != 'application/pdf':
        return jsonify({
            "success": False,
            "message": "Unsupported file type. Only PDF files are allowed.",
            "code": 415
        }), 415

    gizmos = request.args.get('gizmos', default=False, type=bool)

    masker = PDFMasker(ImageMasker(cascade_classifiers, detection_model, image_processor))
    result = masker.mask_file(file, should_draw_gizmos=gizmos)

    return send_file(
        io.BytesIO(result),
        mimetype=file.content_type,
        as_attachment=True,
        download_name=file.filename
    )


@masking_blueprint.route('/image', methods=['POST'])
def mask_image():
    """
    Endpoint for uploading an image file to be masked.
    The image is masked by detecting faces in the image and replacing them with black rectangles.
    Supported image types are JPEG, PNG, and GIF.
    :return: The masked image file.
    :raises: HTTP 400 if no file is uploaded.
    :raises: HTTP 415 if the file type is not supported.
    """
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"success": False, "message": "No file uploaded.", "code": 400}), 400

    file = request.files['file']

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return jsonify({
            "success": False,
            "message": "Unsupported file type. Only JPEG, PNG, and GIF images are allowed.",
            "code": 415
        }), 415

    gizmos = request.args.get('gizmos', default=False, type=bool)

    try:
        masker = ImageMasker(cascade_classifiers, detection_model, image_processor)
        result, _ = masker.mask_file(file, should_draw_gizmos=gizmos)
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "code": 500}), 500

    return send_file(
        io.BytesIO(result),
        mimetype=file.content_type,
        as_attachment=True,
        download_name=file.filename
    )
