import os

def concat_pth_tar():
    dirname = os.path.dirname(__file__)
    model_parts_path = os.path.join(dirname, 'model_parts')
    output_path = os.path.join(dirname, 'modelbest.pth.tar')

    with open(output_path, 'wb') as f:
        lst = [os.path.join(model_parts_path, f.name) for f in os.scandir(model_parts_path) if not f.is_dir()]
        for fname in lst:
            with open(fname,'rb') as g:
                f.write(g.read())