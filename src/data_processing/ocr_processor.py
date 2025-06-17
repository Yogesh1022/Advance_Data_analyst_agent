from PIL import Image
import cv2
import numpy as np

class OCRProcessor:
    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Applies common preprocessing steps to an image for improved OCR accuracy.
        Args:
            image (PIL.Image.Image): The input image.
        Returns:
            PIL.Image.Image: The preprocessed image.
        """
        # Convert PIL Image to OpenCV format (numpy array)
        img_np = np.array(image)
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 1. Skew Correction (using deskew library or manual rotation detection)
        # This is a complex step, often requiring a dedicated library like deskew or image processing algorithms
        # For simplicity, we'll skip full deskewing here, but it's crucial for real applications.
        # Example placeholder:
        # from deskew import determine_skew
        # angle = determine_skew(gray)
        # (h, w) = gray.shape
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # 2. Noise Removal (e.g., Gaussian blur)
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21) # Example denoising

        # 3. Binarization/Thresholding (Otsu's method)
        _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Image Scaling (if necessary, ensure DPI > 300)
        # This depends on the original image resolution. If it's too low, scaling up can help.
        # For example, if img.info.get('dpi') is low, resize.
        # current_dpi = image.info.get('dpi', (72, 72))
        # if current_dpi < 300:
        #     scale_factor = 300 / current_dpi
        #     binarized = cv2.resize(binarized, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Convert back to PIL Image
        preprocessed_image = Image.fromarray(binarized)

        return preprocessed_image

    # Other potential methods:
    # def crop_image(self, image: Image.Image, bbox: tuple) -> Image.Image:
    #     """Crops an image based on a bounding box."""
    #     return image.crop(bbox)

    # def apply_thinning_skeletonization(self, image: Image.Image) -> Image.Image:
    #     """Applies thinning/skeletonization (more advanced, for specific fonts/handwriting)"""
    #     # This typically involves morphological operations and is more complex.
    #     pass