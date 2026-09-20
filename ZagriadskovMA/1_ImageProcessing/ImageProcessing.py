# import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod
import os
from PIL import Image
import argparse
import struct
import zlib
import sys


def cli_argument_parser() -> tuple:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_dir_name = "\\img\\"
    parser = argparse.ArgumentParser(description="Консольная утилита для применения фильтров к изображениям.")
    parser.add_argument("-i", "--input", required=True, help="Путь к входному изображению")
    parser.add_argument("-o", "--output", required=True, help="Путь к выходному изображению")
    parser.add_argument(
        "-f", "--filter", 
        required=True, 
        choices=["resize", "grayscale", "antique", "fadecolor", "tape", "matte", "noize", "neon"],
        help="Название применяемого фильтра"
    )
    parser.add_argument(
        "-p", "--params", 
        nargs="*", 
        default=[], 
        help="Параметры фильтра в формате key=value (например: width=800 height=600 или intensity=0.8)"
    )
    parser.add_argument(
        "--absolute", 
        action="store_true", 
        help="Флаг: указывает, что переданные пути являются абсолютными (по умолчанию - относительные)"
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    if args.absolute:
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
    else:
        input_path = dir_path+img_dir_name+input_path
        output_path = dir_path+img_dir_name+output_path
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл не найден по пути '{input_path}'", file=sys.stderr)
        sys.exit(1)

    filter_kwargs = {}
    for item in args.params:
        if '=' in item:
            key, value = item.split('=', 1)
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            filter_kwargs[key] = value

    return input_path, output_path, args.filter, filter_kwargs

# def read_image(path_to_image: str) -> np.ndarray:
#     with Image.open(path_to_image) as img:
#         return np.array(img)

def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c

def read_image(path_to_image: str) -> np.ndarray:
    with open(path_to_image, 'rb') as f:
        data = f.read()

    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Файл не является валидным PNG")

    offset = 8
    width = height = bit_depth = color_type = 0
    idat_chunks = []

    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        offset += 12 + length

        if chunk_type == b'IHDR':
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', chunk_data)
            if bit_depth != 8:
                raise NotImplementedError("Поддерживается только 8-битный цвет")
            if interlace != 0:
                raise NotImplementedError("Чересстрочный PNG (Adam7) не поддерживается")
        elif chunk_type == b'IDAT':
            idat_chunks.append(chunk_data)
        elif chunk_type == b'IEND':
            break

    channels_map = {0: 1, 2: 3, 6: 4}
    if color_type not in channels_map:
        raise NotImplementedError(f"Цветовой тип {color_type} не поддерживается")
    
    bpp = channels_map[color_type] 
    stride = width * bpp 

    compressed_data = b''.join(idat_chunks)
    decompressed = zlib.decompress(compressed_data)

    recon_bytes = bytearray(height * stride)
    
    for i in range(height):
        row_start_in = i * (1 + stride)
        filter_type = decompressed[row_start_in]
        
        row_start_out = i * stride
        prev_row_start_out = (i - 1) * stride

        for j in range(stride):
            filt_byte = decompressed[row_start_in + 1 + j]

            left = recon_bytes[row_start_out + j - bpp] if j >= bpp else 0
            up = recon_bytes[prev_row_start_out + j] if i > 0 else 0
            up_left = recon_bytes[prev_row_start_out + j - bpp] if (i > 0 and j >= bpp) else 0

            if filter_type == 0:
                recon = filt_byte
            elif filter_type == 1:
                recon = (filt_byte + left) & 0xFF
            elif filter_type == 2:
                recon = (filt_byte + up) & 0xFF
            elif filter_type == 3:
                recon = (filt_byte + (left + up) // 2) & 0xFF
            elif filter_type == 4:
                recon = (filt_byte + paeth_predictor(left, up, up_left)) & 0xFF
            else:
                raise ValueError(f"Неизвестный тип фильтра PNG: {filter_type}")

            recon_bytes[row_start_out + j] = recon

    img_array = np.frombuffer(recon_bytes, dtype=np.uint8)

    if bpp == 1:
        return img_array.reshape((height, width))
    else:
        return img_array.reshape((height, width, bpp))


def write_image(path_to_image: str, image: np.ndarray) -> None:
    img = Image.fromarray(image)
    img.save(path_to_image)


class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> 'ImageFilter':
        filters = {
            "resize": Resize,
            "grayscale": RGB2Grayscale,
            "antique": Antique,
            "fadecolor": FadeColor,
            "tape": Tape,
            "matte": Matte,
            "noize": Noize,
            "neon": Neon
        }

        filter_name = filter_type.lower()
        if filter_name not in filters:
            raise ValueError(f"Неизвестный тип фильтра: '{filter_type}'. "
                             f"Доступные: {list(filters.keys())}")

        return filters[filter_name](**kwargs)



class Resize(ImageFilter):
    def __init__(self, width: int, height: int):
        self.target_width = width
        self.target_height = height

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        orig_height, orig_width = image.shape[:2]
        scale_x = orig_width / self.target_width
        scale_y = orig_height / self.target_height
        has_channels = (len(image.shape) == 3)
        if not has_channels:
            image = image[:, :, np.newaxis]

        x_out = np.arange(self.target_width, dtype=np.float32)
        y_out = np.arange(self.target_height, dtype=np.float32)
        x_src = (x_out + 0.5) * scale_x - 0.5
        y_src = (y_out + 0.5) * scale_y - 0.5
        x1 = np.clip(np.floor(x_src).astype(np.int32), 0, orig_width - 1)
        x2 = np.clip(np.floor(x_src).astype(np.int32) + 1, 0, orig_width - 1)
        y1 = np.clip(np.floor(y_src).astype(np.int32), 0, orig_height - 1)
        y2 = np.clip(np.floor(y_src).astype(np.int32) + 1, 0, orig_height - 1)

        dx = x_src - x1
        dy = y_src - y1
        dx = dx[np.newaxis, :, np.newaxis]
        dy = dy[:, np.newaxis, np.newaxis]

        p_top_left     = image[y1[:, None], x1[None, :]]
        p_top_right    = image[y1[:, None], x2[None, :]]
        p_bottom_left  = image[y2[:, None], x1[None, :]]
        p_bottom_right = image[y2[:, None], x2[None, :]]

        top = p_top_left * (1 - dx) + p_top_right * dx
        bottom = p_bottom_left * (1 - dx) + p_bottom_right * dx
        result = top * (1 - dy) + bottom * dy
        result = np.clip(result, 0, 255).astype(np.uint8)

        if not has_channels:
            result = result.squeeze(axis=-1)

        return result


class RGB2Grayscale(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image.copy()
        
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return gray.astype(np.uint8)

class Antique(ImageFilter):
    def __init__(self, intensity: float = 0.7):
        self.intensity = max(0.0, min(1.0, float(intensity)))
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image_rgb = np.stack((image,)*3, axis=-1)
        else:
            image_rgb = image[..., :3]

        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ], dtype=np.float32)

        antique_image = np.dot(image_rgb.astype(np.float32), sepia_matrix.T)
        if self.intensity < 1.0:
            result = image_rgb.astype(np.float32) * (1.0 - self.intensity) + antique_image * self.intensity
        else:
            result = antique_image

        result = np.clip(result, 0, 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 4:
            result = np.dstack((result, image[:, :, 3]))

        return result

class FadeColor(ImageFilter):
    def __init__(self, max_fade: float = 50.0):
        self.max_fade = max_fade

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32)
        f_min = self.max_fade
        f_max = 255.0 - self.max_fade

        scale = (f_max - f_min) / 255.0

        if len(image.shape) == 3 and image.shape[2] == 4:
            rgb = result[:, :, :3]
            faded_rgb = f_min + rgb * scale
            result[:, :, :3] = faded_rgb
        else:
            result = f_min + result * scale

        return np.clip(result, 0, 255).astype(np.uint8)

class Tape(ImageFilter):
    def __init__(self, mode: str = 'infrared', grain_amount: float = 0.3):
        self.mode = mode.lower()
        self.grain_amount = max(0.0, min(1.0, float(grain_amount)))

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image_rgb = np.stack((image,)*3, axis=-1).astype(np.float32)
        else:
            image_rgb = image[..., :3].astype(np.float32)

        if self.mode == 'infrared':
            color_matrix = np.array([
                [0.1,  1.5, -0.2],
                [0.6,  0.2,  0.1],
                [-0.1, 0.1,  0.8]
            ], dtype=np.float32)
        elif self.mode == 'classic':
            color_matrix = np.array([
                [0.9,  0.1,  0.1],
                [0.0,  0.85, 0.15],
                [0.1,  0.1,  0.7]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Неизвестный параметр фильтра: '{self.mode}'. Доступны 'classic' и 'infraded'.")

        film_colored = np.dot(image_rgb, color_matrix.T)

        if self.grain_amount > 0.0:
            h, w, c = film_colored.shape
            std_dev = self.grain_amount * 25.0
            grain = np.random.normal(loc=0.0, scale=std_dev, size=(h, w, 1))

            luminance = (0.299 * film_colored[..., 0] + 0.587 * film_colored[..., 1] + 0.114 * film_colored[..., 2]) / 255.0
            grain_mask = 1.0 - 4.0 * (luminance - 0.5) ** 2
            grain_mask = np.clip(grain_mask[..., None], 0.1, 1.0)
            film_colored += grain * grain_mask

        result = film_colored
        result = np.clip(result, 0, 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 4:
            result = np.dstack((result, image[:, :, 3]))

        return result

class Matte(ImageFilter):
    def __init__(self, scale: float = 0.9, feather: float = 0.15, fill_color: tuple = (255, 255, 255)):
        self.scale = max(0.1, min(1.0, float(scale)))
        self.feather = max(0.001, min(0.5, float(feather)))
        self.fill_color = np.array(fill_color, dtype=np.float32)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        center_y, center_x = h / 2.0, w / 2.0
        radius_x = center_x * self.scale
        radius_y = center_y * self.scale
        y_indices = np.arange(h, dtype=np.float32)
        x_indices = np.arange(w, dtype=np.float32)
        delta_x = (x_indices[None, :] - center_x) / radius_x
        delta_y = (y_indices[:, None] - center_y) / radius_y
        R = np.sqrt(delta_x ** 2 + delta_y ** 2)

        low_bound = 1.0 - self.feather
        high_bound = 1.0 + self.feather
        mask = (R - low_bound) / (high_bound - low_bound)
        mask = np.clip(mask, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        mask = mask[..., None]

        if len(image.shape) == 2:
            image_float = image.astype(np.float32)[..., None]
            fill_bg = np.array([self.fill_color.mean()], dtype=np.float32)
        else:
            image_float = image[..., :3].astype(np.float32)
            fill_bg = self.fill_color
        result_rgb = image_float * (1.0 - mask) + fill_bg * mask
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 4:
            return np.dstack((result_rgb, image[:, :, 3]))

        return result_rgb.squeeze() if len(image.shape) == 2 else result_rgb

class Noize(ImageFilter):
    def __init__(
        self, noise_amount: float = 0.008, num_scratches: int = 12, scratch_intensity: float = 0.7):
        self.noise_amount = max(0.0, min(0.1, float(noise_amount)))
        self.num_scratches = max(0, int(num_scratches))
        self.scratch_intensity = max(0.0, min(1.0, float(scratch_intensity)))

    def _draw_line(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int = 1):
        h, w = mask.shape
        num_points = int(np.hypot(x1 - x0, y1 - y0)) + 1
    
        if num_points <= 1:
            return

        t = np.linspace(0, 1, num_points)
        x_pts = x0 + t * (x1 - x0) + np.random.normal(0, 0.3, num_points)
        y_pts = y0 + t * (y1 - y0) + np.random.normal(0, 0.3, num_points)

        line_alpha = np.random.uniform(0.4, 1.0, size=num_points)

        radius = max(0.8, thickness / 1.5)

        for i in range(num_points):
            cx, cy = x_pts[i], y_pts[i]
        
            r_int = int(np.ceil(radius)) + 1
            x_min, x_max = max(0, int(cx - r_int)), min(w, int(cx + r_int + 1))
            y_min, y_max = max(0, int(cy - r_int)), min(h, int(cy + r_int + 1))
        
            if x_min >= x_max or y_min >= y_max:
                continue

            grid_y, grid_x = np.ogrid[y_min:y_max, x_min:x_max]
            dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)

            falloff = np.clip(1.0 - (dist / (radius + 0.5)), 0.0, 1.0)

            mask[y_min:y_max, x_min:x_max] = np.maximum(
                mask[y_min:y_max, x_min:x_max], 
                falloff * line_alpha[i]
            )

    def _generate_scratches_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(self.num_scratches):
            if np.random.rand() < 0.8:
                x0 = np.random.randint(0, w)
                y0 = np.random.randint(0, int(h * 0.3))
                x1 = x0 + np.random.randint(-15, 15)
                y1 = y0 + np.random.randint(int(h * 0.4), h)
            else:
                x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
                x1 = x0 + np.random.randint(-40, 40)
                y1 = y0 + np.random.randint(-40, 40)

            thickness = 1 if np.random.rand() > 0.2 else 2
            self._draw_line(mask, x0, y0, x1, y1, thickness=thickness)

        return mask


    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        if self.noise_amount > 0:
            rand_matrix = np.random.rand(h, w)

            dust_white_mask = rand_matrix < (self.noise_amount * 0.6)
            dust_black_mask = (rand_matrix >= (self.noise_amount * 0.6)) & (rand_matrix < self.noise_amount)

            if len(image.shape) == 3:
                dust_white_mask = dust_white_mask[..., None]
                dust_black_mask = dust_black_mask[..., None]

            result[dust_white_mask.repeat(result.shape[-1] if len(image.shape) == 3 else 1, axis=-1 if len(image.shape) == 3 else 0)] = 245.0
            result[dust_black_mask.repeat(result.shape[-1] if len(image.shape) == 3 else 1, axis=-1 if len(image.shape) == 3 else 0)] = 15.0

        if self.num_scratches > 0 and self.scratch_intensity > 0:
            scratches_mask = self._generate_scratches_mask(h, w)
            if scratches_mask.max() > 0:
                padded = np.pad(scratches_mask, ((1, 1), (1, 1)), mode='edge')
                scratches_mask = (
                    padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:] +
                    padded[1:-1, 1:-1] * 2.0
                ) / 6.0

            scratches_mask = (scratches_mask * self.scratch_intensity * 180.0)[..., None]
            result[..., :3] = np.clip(result[..., :3] + scratches_mask, 0, 255)

        return np.clip(result, 0, 255).astype(np.uint8)

    
class Neon(ImageFilter):
    def __init__(self, glow_color: tuple = (0, 255, 230), intensity: float = 1.5, blur_radius: int = 2):
        self.glow_color = np.array(glow_color, dtype=np.float32)
        self.intensity = float(intensity)
        self.blur_radius = max(1, int(blur_radius))

    def _sobel_edges(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        
        padded = np.pad(gray, ((1, 1), (1, 1)), mode='edge')

        gx = (
            -1 * padded[:-2, :-2] + 1 * padded[:-2, 2:] +
            -2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] +
            -1 * padded[2:, :-2] + 1 * padded[2:, 2:]
        )
        
        gy = (
            -1 * padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - 1 * padded[:-2, 2:] +
             1 * padded[2:, :-2] + 2 * padded[2:, 1:-1] + 1 * padded[2:, 2:]
        )

        magnitude = np.hypot(gx, gy)
        return magnitude

    def _simple_blur(self, image: np.ndarray, radius: int) -> np.ndarray:
        blurred = image.copy()
        for _ in range(radius):
            padded = np.pad(blurred, ((1, 1), (1, 1), (0, 0)), mode='edge')
            blurred = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] +
                padded[1:-1, :-2] + padded[1:-1, 2:] +
                padded[1:-1, 1:-1]  
            ) / 5.0
        return blurred

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            orig_rgb = np.stack((image,)*3, axis=-1).astype(np.float32)
        else:
            orig_rgb = image[..., :3].astype(np.float32)

        gray = 0.299 * orig_rgb[..., 0] + 0.587 * orig_rgb[..., 1] + 0.114 * orig_rgb[..., 2]
        edges = self._sobel_edges(gray)
        edges = (edges / (edges.max() + 1e-5)) * self.intensity
        edges = np.where(edges > 0.15, edges, 0.0)
        edges = np.clip(edges, 0.0, 1.0)[..., None]

        sharp_neon = edges * self.glow_color
        glow_aura = self._simple_blur(sharp_neon, radius=self.blur_radius * 2)
        final_neon = sharp_neon + glow_aura * 1.2
        result = orig_rgb * 0.7 + final_neon

        result = np.clip(result, 0, 255).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 4:
            result = np.dstack((result, image[:, :, 3]))

        return result


def main():
    try:
        input_path, output_path, filter, filter_kwargs = cli_argument_parser()
        image = read_image(input_path)
    except Exception as e:
        print(f'Ошибка при вводе данных: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        filter_instance: ImageFilter = ImageFilter.get_filter(filter, **filter_kwargs)
        result_image = filter_instance.apply_filter(image)
    except Exception as e:
        print(f'Ошибка при обработке изображения: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        write_image(output_path, result_image)
    except Exception as e:
        print(f'Ошибка при сохранении изображения: {e}', file=sys.stderr)
        sys.exit(1)


if __name__=="__main__":
    main()
