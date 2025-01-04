def read_file_content(file):
    if file is None:
        return ""
    with open(file.name, 'r', encoding='utf-8') as f:
        return f.read()
