import os
import shutil


def create_empty_dir(dir_path, remove_if_exists=False, yes_to_all=False):
    try:
        if remove_if_exists:
            if not safe_remove_dirs(dir_path, yes_to_all=yes_to_all):
                return False
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        pass

    return True


def safe_remove_dirs(dir_path, yes_to_all=False):
    if not os.path.exists(dir_path):
        return True
    if not yes_to_all:
        ans = input(f"Remove dirs in {dir_path} [y/N]?")
    else:
        ans = "y"
    if ans.lower() == "y":
        print(f"Remove dirs {dir_path}")
        shutil.rmtree(dir_path)
        return True
    else:
        return False


def recurse_get_filepath_in_dir(dir_path, suffix=None, prefix=None, followlinks=False):
    if suffix is not None:
        suffix = suffix.split(";;")
    if prefix is not None:
        prefix = prefix.split(";;")

    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good

    res = []
    for dir_path, _, files in os.walk(dir_path, followlinks=followlinks):
        for file in files:
            if suffix is not None or prefix is not None:
                if check_file(file):
                    res.append(os.path.join(dir_path, file))
            else:
                res.append(os.path.join(dir_path, file))
    res.sort()
    return res
