from aiohttp import web
from pyfefi2 import compress, Slice
from pyfefi2.dataset import open_dataset
from pyfefi2.datafolder import LocalFolder
import numpy as np
import netCDF4
import os
import time
from pathlib import Path

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

        # check time stamp
        try:
            if abs(time.time() - int(timestamp)) > 30:
                return web.Response(text="Request expired", status=401)
        except ValueError:
            return web.Response(text="Invalid timestamp", status=400)

        # check user
        with open(self.key_path, 'r') as f:
            trusted_keys = json.load(f)

        users = trusted_keys.get('users', [])
        matched_users = [u for u in users if u.get('name') == user_id]

        if not matched_users:
            return web.Response(text='Unknown user', status=403)

        # verify signature
        for user in matched_users:
            keys = list(user.get('keys', []))
            if 'key' in user:
                keys.append(user['key'])

            for key_str in keys:
                try:
                    pub_key_bytes = base64.b64decode(key_str)
                    public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_key_bytes)

                    signature = base64.b64decode(signature_b64)
                    public_key.verify(signature, (timestamp + user_id).encode('utf-8'))

                    request['user'] = user
                    request['config'] = trusted_keys
                    return None
                except (InvalidSignature, ValueError, TypeError, Exception):
                    pass

        return web.Response(text="Invalid signature", status=403)


    async def send_data(self, request):
        auth_res = await self.auth(request)
        if auth_res is not None:
            return auth_res

        fn = request.path[len(self.url_path):]
        full_fn = os.path.join(self.base, fn)
        if not Path(full_fn).is_relative_to(Path(self.base)):
            return web.HTTPForbidden(text="Accessing forbidden directory")

        # Access control
        user = request['user']
        config = request['config']

        req_path_str = os.path.normpath(fn.lstrip('/'))
        if req_path_str == '.':
            req_path_str = ''
        req_path = Path(req_path_str)

        # Collect rules
        rules = [user]
        groups = config.get('groups', {})
        for g in user.get('groups', []):
            if g in groups:
                rules.append(groups[g])

        is_whitelist_mode = any(r.get('mode') == 'whitelist' for r in rules)

        whitelist_folders = []
        blacklist_folders = []
        for r in rules:
            if r.get('mode') == 'whitelist':
                whitelist_folders.extend(r.get('folders', []))
            elif r.get('mode') == 'blacklist':
                blacklist_folders.extend(r.get('folders', []))

        # Check blacklist first
        for folder in blacklist_folders:
            if folder == '/' or folder == '':
                folder_path = Path('')
            else:
                folder_path = Path(folder)
            # If the requested path is inside the blacklisted folder, or exactly the blacklisted folder
            if req_path == folder_path or req_path.is_relative_to(folder_path):
                return web.HTTPForbidden(text="Access denied")

        # Then check whitelist if in whitelist mode
        if is_whitelist_mode:
            allowed = False
            for folder in whitelist_folders:
                if folder == '/' or folder == '':
                    folder_path = Path('')
                else:
                    folder_path = Path(folder)
                # If requested path is inside the whitelisted folder, it's allowed
                # If the requested path is a parent of the whitelisted folder (e.g. asking for root, when 'secret' is whitelisted),
                # we should probably allow access to list, but reading actual files inside that parent directory that aren't whitelisted will be caught later?
                # Actually, wait. If they request the root directory 'list', and 'secret' is whitelisted, they need to be able to list root.
                if req_path == folder_path or req_path.is_relative_to(folder_path) or folder_path.is_relative_to(req_path):
                    allowed = True
                    break
            if not allowed:
                return web.HTTPForbidden(text="Access denied")

        query = request.query.get('q')

        if query == 'raw':
            return web.FileResponse(os.path.join(full_fn))
        elif query == 'list':
            try:
                fns = os.listdir(full_fn)
                # Filter out blacklisted folders from list if they are directly inside req_path
                filtered_fns = []
                for f in fns:
                    item_path = req_path / f

                    is_blacklisted = False
                    for bfolder in blacklist_folders:
                        if bfolder == '/' or bfolder == '':
                            bfolder_path = Path('')
                        else:
                            bfolder_path = Path(bfolder)
                        if item_path == bfolder_path or item_path.is_relative_to(bfolder_path):
                            is_blacklisted = True
                            break

                    if is_blacklisted:
                        continue

                    if is_whitelist_mode:
                        is_whitelisted = False
                        for wfolder in whitelist_folders:
                            if wfolder == '/' or wfolder == '':
                                wfolder_path = Path('')
                            else:
                                wfolder_path = Path(wfolder)
                            if item_path == wfolder_path or item_path.is_relative_to(wfolder_path) or wfolder_path.is_relative_to(item_path):
                                is_whitelisted = True
                                break
                        if not is_whitelisted:
                            continue

                    filtered_fns.append(f)

                return web.json_response({'files': filtered_fns})
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
                        data = ds[name, slc]
                        if np.isscalar(data) or data.ndim == 0:
                            return web.Response(body=data.tobytes())
                        else:
                            data = compress.compress_array(data)
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
