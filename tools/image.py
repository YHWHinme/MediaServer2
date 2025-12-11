from pillow import PIL

def convert_to_base64(pil_image):
    """Convert PIL image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
