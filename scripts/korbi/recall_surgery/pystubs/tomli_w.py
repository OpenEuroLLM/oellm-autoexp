"""Minimal stand-in for tomli_w (not in the training image), imported by titan_oellm's metrics module, which the
native exporter pulls in through lm_scaling.titan_ext. Last on PYTHONPATH: a real tomli_w wins. Writes flat and
one-level-nested tables of str / int / float / bool / lists thereof."""


def _val(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_val(x) for x in v) + ']'
    s = str(v).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{s}"'


def dumps(obj, **_kw):
    lines, tables = [], []
    for k, v in obj.items():
        (tables if isinstance(v, dict) else lines).append((k, v))
    out = [f'{k} = {_val(v)}' for k, v in lines]
    for k, t in tables:
        out.append(f'\n[{k}]')
        out.extend(f'{kk} = {_val(vv)}' for kk, vv in t.items() if not isinstance(vv, dict))
    return '\n'.join(out) + '\n'


def dump(obj, fp, **kw):
    fp.write(dumps(obj, **kw).encode())
