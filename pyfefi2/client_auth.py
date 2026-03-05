import base64
import time
import base64
import requests
import os
import platform
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

def get_key_path():
    env_path = os.getenv('PYFEFI_KEY_PATH')
    dir_name = 'pyfefi2'

    if env_path:
        target_path = Path(env_path)
    else:
        home = Path.home()
        system = platform.system()

        if system == 'Windows':
            appdata = os.getenv('APPDATA', home/'AppData'/'Roaming')
            target_path = Path(appdata) / dir_name
        elif system == 'Darwin':
            target_path = home / "Library" / "Application Support" / dir_name
        else:
            xdg_config = os.getenv("XDG_CONFIG_HOME", home / ".config")
            target_path = Path(xdg_config) / dir_name

    target_path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target_path, 0o700)
    except OSError:
        pass

    return target_path

class Auth:

    def __init__(self):
        key_dir = get_key_path()
        key_file = (key_dir / 'private_key.pem').absolute()
        user_id_file = (key_dir / 'user').absolute()

        if not os.path.exists(user_id_file) or not os.path.exists(key_file):
            # Generate keys
            user = os.getenv('PYFEFI_USER')
            if not user:
                print('No credential found. Generating new credential...')
                user = input('Input your username:\n')
            with open(user_id_file, 'w') as f:
                f.write(user)

            key = ed25519.Ed25519PrivateKey.generate()
            pem = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(key_file, 'wb') as f:
                f.write(pem)

            self.user = user
            self.key = key

            print('\nCredential saved.')
            print('Please send the following information to the server maintainer.\n')
            self.print()
            input("\nPress the Enter key to continue.")
        else:
            with open(user_id_file, 'r') as f:
                user = f.read()
            with open(key_file, 'rb') as f:
                key = serialization.load_pem_private_key(f.read(), password=None)
            self.user = user
            self.key = key

    def get_headers(self):
        timestamp = str(int(time.time()))
        sig = self.key.sign((timestamp + self.user).encode('utf-8'))
        sig_b64 = base64.b64encode(sig).decode('utf-8')

        headers = {
            'X-Identity': self.user,
            'X-Timestamp': timestamp,
            'X-Signature': sig_b64
        }
        return headers

    def print(self):
        pubkey = base64.b64encode(
                self.key.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
                )).decode('utf-8')
        print(f'Username: {self.user}')
        print(f'Public key: {pubkey}')
