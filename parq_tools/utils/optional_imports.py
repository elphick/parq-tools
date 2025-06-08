def get_tqdm():
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        def dummy(iterable, *args, **kwargs):
            return iterable
        return dummy

