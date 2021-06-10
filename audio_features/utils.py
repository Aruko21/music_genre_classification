import os


def get_all_directories(dirpath):
    return [o for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, o))]


def get_audios_by_genre(root, genre):
    return list(os.path.join(root, genre, o) for o in os.listdir(os.path.join(root, genre)))
