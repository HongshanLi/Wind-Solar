import os

def path_check(path):
    """ Make sure the path exists."""

    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise