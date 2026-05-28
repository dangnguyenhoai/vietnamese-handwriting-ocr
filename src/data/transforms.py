import random
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


class HandwritingAugment:
    """
    Augmentation nhẹ cho OCR chữ viết tay dòng.
    Không được phá layout quá mạnh, vì CTC cần thứ tự ký tự từ trái sang phải.
    """

    def __init__(
        self,
        p_rotate=0.35,
        p_shear=0.25,
        p_brightness=0.35,
        p_contrast=0.35,
        p_blur=0.15,
        p_noise=0.20,
        p_invert=0.00,
    ):
        self.p_rotate = p_rotate
        self.p_shear = p_shear
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_invert = p_invert

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")

        if random.random() < self.p_rotate:
            angle = random.uniform(-2.5, 2.5)
            image = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                expand=True,
                fillcolor=255,
            )

        if random.random() < self.p_shear:
            image = self.random_shear(image)

        if random.random() < self.p_brightness:
            factor = random.uniform(0.75, 1.25)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() < self.p_contrast:
            factor = random.uniform(0.75, 1.35)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() < self.p_blur:
            radius = random.uniform(0.2, 0.7)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        if random.random() < self.p_noise:
            image = self.add_pixel_noise(image)

        if random.random() < self.p_invert:
            image = ImageOps.invert(image)

        return image

    def random_shear(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        shear = random.uniform(-0.08, 0.08)

        xshift = abs(shear) * height
        new_width = width + int(round(xshift))

        if shear > 0:
            matrix = (1, shear, -xshift, 0, 1, 0)
        else:
            matrix = (1, shear, 0, 0, 1, 0)

        image = image.transform(
            (new_width, height),
            Image.Transform.AFFINE,
            matrix,
            resample=Image.Resampling.BILINEAR,
            fillcolor=255,
        )

        return image

    def add_pixel_noise(self, image: Image.Image) -> Image.Image:
        """
        Noise rất nhẹ. Không dùng noise nặng kiểu phá chữ.
        """
        pixels = image.load()
        width, height = image.size

        noise_ratio = random.uniform(0.001, 0.004)
        num_pixels = int(width * height * noise_ratio)

        for _ in range(num_pixels):
            x = random.randrange(width)
            y = random.randrange(height)
            delta = random.randint(-25, 25)
            pixels[x, y] = max(0, min(255, pixels[x, y] + delta))

        return image


def build_train_transform():
    return HandwritingAugment()


def build_val_transform():
    return None