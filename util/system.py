__author__ = 'matt'

from getpass import getuser
from os import listdir
from os.path import isfile, join


def get_system_username():
    # Wrapped to avoid imports plus to fancier things later
    return getuser()


def get_files_in_dir(directory, ignore_hidden=True):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    return [f for f in files if not f.startswith(".")] if ignore_hidden else files


def load_json(json_path):
    from json import loads

    if str(json_path) == '':
        return None

    fp = open(json_path, 'r')
    raw_data = [x for x in fp.readlines()]
    fp.close()
    raw_str = ''.join(raw_data)
    cln_str = raw_str.replace('\n', '')

    return loads(cln_str)
