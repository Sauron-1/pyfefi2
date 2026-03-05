from abc import abstractmethod
import os
import warnings
import requests

from .client_auth import Auth

class DataFolder:

    @abstractmethod
    def type(self):
        raise NotImplementedError()

    @abstractmethod
    def refresh(self):
        raise NotImplementedError()

    @abstractmethod
    def list(self):
        raise NotImplementedError()

    @abstractmethod
    def exists(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def has_data(self, prefix, frame):
        raise NotImplementedError()

    @abstractmethod
    def read_file(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def write_file(self, fn, data):
        raise IOError(f"Can't write file {fn}: write not implemented.")

class LocalFolder(DataFolder):

    def __init__(self, path):
        if os.path.isfile(path):
            raise RuntimeError(f'`path` must be a folder, got {path}')
        self.path = os.path.abspath(path)

    def type(self):
        return 'local'

    def refresh(self):
        pass

    def list(self):
        return os.listdir(self.path)

    def exists(self, fn):
        return os.path.exists(os.path.join(self.path, fn))

    def has_data(self, prefix, frame):
        base = os.path.join(self.path, prefix + f'{frame:05d}')
        return os.path.exists(base + '.nc') or os.path.exists(base + '.sz3')

    def read_file(self, fn):
        with open(os.path.join(self.path, fn), 'r') as f:
            return f.read()

    def write_file(self, fn, data):
        target_file = os.path.join(self.path, fn)
        with open(target_file, 'w') as f:
            f.write(data)

class HttpFolder(DataFolder):

    def __init__(self, path: str):
        if not path.endswith('/'):
            path = path + '/'
        self.path = path
        self.auth = Auth()
        self.files = []

    def type(self):
        return 'http'

    def refresh(self):
        resp = requests.get(
            self.path,
            headers = self.auth.get_headers(),
            params = dict(q='list')
        )
        self.files = resp.json()['files']

    def list(self):
        return set(self.files)

    def exists(self, fn):
        return fn in self.files

    def has_data(self, prefix, frame):
        base = prefix + f'{frame:05d}'
        return self.exists(base + '.nc') or self.exists(base + '.sz3')

    def read_file(self, fn):
        resp = requests.get(
            self.path + fn,
            headers = self.auth.get_headers(),
            params = dict(q='raw')
        )
        return resp.text

    def write_file(self, fn, data):
        raise NotImplementedError("Can't write file to HTTP folder")

def open_folder(path):
    if path.startswith('http') and not os.path.exists(path):
        return HttpFolder(path)
    return LocalFolder(path)
