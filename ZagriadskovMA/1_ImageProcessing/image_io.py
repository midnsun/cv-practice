import struct
import zlib
from PIL import Image
import numpy as np

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
