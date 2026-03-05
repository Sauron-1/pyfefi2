from aiohttp import web
from pyfefi2 import compress, Slice
from pyfefi2.dataset import open_dataset
from pyfefi2.datafolder import LocalFolder
import numpy as np
import netCDF4
import os
import time

import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
import json

def decode_index(s):
    """Decodes a string back into a tuple of slices/integers."""
    if not s:
        return ()

    decoded = []
    for part in s.split(","):
        if ":" in part:
            # Reconstruct the slice
            # split(':') gives 1 to 3 parts
            slice_parts = [(int(p) if p != "" else None) for p in part.split(":")]
            # Ensure we have exactly 3 parts for the slice constructor
            while len(slice_parts) < 3:
                slice_parts.append(None)
            decoded.append(slice(*slice_parts))
        else:
            decoded.append(int(part))
    return tuple(decoded)

class Handler:

    def __init__(self, base, path='/', trusted_keys_path=None):
        self.base = base
        if not self.base.endswith('/'):
            self.base = self.base + '/'
        if trusted_keys_path is None:
            trusted_keys_path = os.environ.get('PYFEFI_TRUSTED_KEY_PATH')
        self.key_path = trusted_keys_path
        self.url_path = path

    def bind(self, app):
        #app.add_routes([web.get(self.url_path + '{fn}', self.send_data)])
        app.router.add_get(self.url_path + '{tail:.*}', self.send_data)
        #app.add_routes([web.get('/{name}', self.send_data)])

    async def auth(self, request):
        user_id = request.headers.get('X-Identity')
        timestamp = request.headers.get('X-Timestamp')
        signature_b64 = request.headers.get('X-Signature')

        # basic check
        if not all([user_id, timestamp, signature_b64]):
            return web.Response(text="Missing auth headers", status=401)

        # check user
        with open(self.key_path, 'r') as f:
            trusted_keys = json.load(f)
        if user_id not in trusted_keys:
            return web.Response(text='Unknown user', status=403)

        # check time stamp
        try:
            if abs(time.time() - int(timestamp)) > 30:
                return web.Response(text="Request expired", status=401)
        except ValueError:
            return web.Response(text="Invalid timestamp", status=400)

        # verify signature
        try:
            pub_key_bytes = base64.b64decode(trusted_keys[user_id])
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_key_bytes)

            signature = base64.b64decode(signature_b64)
            public_key.verify(signature, (timestamp + user_id).encode('utf-8'))

        except (InvalidSignature, Exception) as e:
            return web.Response(text="Invalid signature", status=403)


    async def send_data(self, request):
        auth_res = await self.auth(request)
        if auth_res is not None:
            return auth_res

        fn = request.path[len(self.url_path):]
        full_fn = os.path.join(self.base, fn)

        query = request.query.get('q')

        if query == 'raw':
            return web.FileResponse(os.path.join(full_fn))
        elif query == 'list':
            try:
                fns = os.listdir(full_fn)
                return web.json_response({'files': fns})
            except Exception:
                return web.HTTPNotFound()
        else:
            try:
                folder = LocalFolder(os.path.dirname(full_fn))
                ds = open_dataset(folder, os.path.basename(full_fn), allow_remote=False)
            except Exception as e:
                return web.HTTPNotFound()
            name = request.query.get('name')

            if query == 'has':
                res_str = 'T' if ds.has_array(name) else 'F'
                return web.Response(text=res_str)
            elif query == 'get':
                try:
                    slc = decode_index(request.query.get('slc'))
                except Exception:
                    return web.HTTPBadRequest(text="Slice format error")
                if name != 'param':
                    try:
                        data = compress.compress_array(ds[name, slc])
                        return web.Response(body=data)
                    except Exception as e:
                        return web.HTTPInternalServerError(text="Failed to read or decompress data")
                else:
                    try:
                        data = ds[name, slc]
                        return web.Response(body=data.tobytes())
                    except Exception as e:
                        return web.HTTPInternalServerError(text="Failed to read params")
            else:
                return web.HTTPBadRequest(text="Unknown method")

def start_server(base, host, port, path='/', key_path=None):
    app = web.Application()
    hdl = Handler(base, path=path, trusted_keys_path=key_path)
    hdl.bind(app)
    web.run_app(app, host=host, port=port)
