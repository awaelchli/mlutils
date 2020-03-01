import os
import re


def makedirs(filename):
    folder, _ = os.path.split(filename)
    os.makedirs(folder, exist_ok=True)


def list_to_file(items, file):
    makedirs(file)
    lines = '\n'.join([str(item) for item in items])
    with open(file, 'w') as f:
        f.writelines(lines)


def list_from_file(file):
    with open(file, 'r') as file:
        items = [f.strip() for f in file.readlines()]
    return items


def add_suffix(filename, suffix):
    name, ext = os.path.splitext(filename)
    return name + suffix + ext


def index_from_filename(filename):
    result = re.findall(r'\d+', str(filename))
    if result:
        return int(result[-1])
