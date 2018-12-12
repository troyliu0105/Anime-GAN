def restore_from_file(filename):
    import pickle
    with open(filename, mode='rb') as f:
        return pickle.load(f)


def save_to_file(filename, params):
    import pickle
    with open(filename, mode='wb') as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    save_to_file('variables', [16923, 5])
