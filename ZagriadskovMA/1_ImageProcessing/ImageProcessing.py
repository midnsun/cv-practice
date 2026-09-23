from cli_parser import cli_argument_parser
from image_io import read_image, write_image
from my_filters import ImageFilter

import sys


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
