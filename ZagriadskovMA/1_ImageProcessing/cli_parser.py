import os
import argparse
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
