import os

def concat_pth_tar(parts_folder_name, output_path):
    dirname = os.path.dirname(__file__)
    model_parts_path = os.path.join(dirname, parts_folder_name)

    output_file = open(output_path, 'wb')
    parts = os.listdir(model_parts_path)
    parts.sort()

    read_size = 8
    for file in parts:
        path = os.path.join(model_parts_path, file)
        input_file = open(path, 'rb')
        while True:
            bytes = input_file.read(read_size)
            if not bytes:
                break
            output_file.write(bytes)
        input_file.close()
    output_file.close()