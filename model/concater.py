import os

def concat_pth_tar():
    dirname = os.path.dirname(__file__)
    model_parts_path = os.path.join(dirname, 'model_parts')
    dest_filename = os.path.join(dirname, 'modelbest.pth.tar')

    output_file = open(dest_filename, 'wb')
    parts = os.listdir(model_parts_path)
    parts.sort()

    for file in parts:
        path = os.path.join(model_parts_path, file)
        input_file = open(path, 'rb')
        while True:
            bytes = input_file.read(8)
            if not bytes:
                break
            output_file.write(bytes)
        input_file.close()
    output_file.close()