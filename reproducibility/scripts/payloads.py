"""Portable exact readers for the publication's deduplicated prediction and TRT packs."""
import argparse
import hashlib
import io
import json
import struct
from pathlib import Path
from zipfile import ZipFile

import zstandard as zstd

MAGIC=b"MJXSEQ01"
FORMATS=("raw.jsonl","adapted.jsonl","coco.json")


def read_exact(f,n):
    parts=[]
    while n:
        b=f.read(n)
        if not b:
            raise ValueError("Truncated canonical payload")
        parts.append(b)
        n-=len(b)
    return b"".join(parts)


def replay_external(infile, outputs=None):
    """Return the exact reconstructed payload hashes, sizes and sample count."""
    fingerprints=[hashlib.sha256() for _ in range(3)]
    sizes=[0,0,0]
    with zstd.ZstdDecompressor().stream_reader(infile,closefd=False) as dec:
        if read_exact(dec,8)!=MAGIC:
            raise ValueError("Unknown external format")
        count,=struct.unpack("<I",read_exact(dec,4))
        if count not in (459,16592):
            raise ValueError("Unexpected frozen split size")
        for _ in range(count):
            lengths=struct.unpack("<III",read_exact(dec,12))
            if max(lengths)>8*1024*1024:
                raise ValueError("Invalid frame length")
            for i,n in enumerate(lengths):
                b=read_exact(dec,n)
                fingerprints[i].update(b)
                sizes[i]+=len(b)
                if outputs is not None and outputs[i] is not None:
                    outputs[i].write(b)
        if dec.read(1):
            raise ValueError("Unexpected trailing bytes")
    return [h.hexdigest().upper() for h in fingerprints],sizes,count


def restore_plane(packed):
    """Restore exact original NPY file, including its native NPY header."""
    import numpy as np
    binary=zstd.ZstdDecompressor().decompress(packed,max_output_size=8000000)
    n,=struct.unpack_from("<H",binary)
    header=binary[2:2+n]
    if not header.startswith(b"\x93NUMPY") or len(binary)!=2+n+7560000:
        raise ValueError("Malformed TRT byte-plane stream")
    x=np.frombuffer(binary,dtype=np.uint8,offset=2+n).reshape(4,5,15,25200)
    return header+np.ascontiguousarray(x.transpose(1,2,0,3)).tobytes()


def raw_tensor(npy):
    """Interpret native byte planes as the archived (5,25200,15) float32 tensor."""
    import numpy as np
    a=np.load(io.BytesIO(npy),allow_pickle=False)
    if a.shape!=(5,15,4,25200) or a.dtype!=np.uint8:
        raise ValueError("Unexpected tensor dtype/shape")
    return np.ascontiguousarray(a.transpose(0,1,3,2)).view("<f4").reshape(5,15,25200).transpose(0,2,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(required=True,dest="type")
    q=sub.add_parser("external",help="Extract exact raw, adapted and COCO streams for a run")
    q.add_argument("asset",type=Path); q.add_argument("member"); q.add_argument("output",type=Path)
    q=sub.add_parser("member",help="Read one canonical Table 14 object; restore NPY if transformed")
    q.add_argument("asset",type=Path); q.add_argument("member"); q.add_argument("output",type=Path)
    a=p.parse_args()
    with ZipFile(a.asset) as z:
        if a.type=="external":
            a.output.mkdir(parents=True,exist_ok=False)
            from contextlib import ExitStack
            with ExitStack() as s:
                bb=[s.enter_context((a.output/n).open("xb")) for n in FORMATS]
                hh,sizes,n=replay_external(s.enter_context(z.open(a.member)),bb)
            print(json.dumps(dict(payload_sha256=hh,bytes=sizes,samples=n)))
        else:
            b=z.read(a.member)
            if a.member.endswith(".planes.zst"):
                b=restore_plane(b)
            with a.output.open("xb") as f:
                f.write(b)
            print(hashlib.sha256(b).hexdigest())


if __name__=="__main__":
    main()
