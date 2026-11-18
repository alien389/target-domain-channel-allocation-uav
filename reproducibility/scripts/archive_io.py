"""Read STORE members of the transport as bounded seekable views, without extraction."""
import hashlib
import io
import re
import struct
from pathlib import PurePosixPath
from zipfile import ZIP_STORED


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(b):
    return hashlib.sha256(b).hexdigest().upper()


def stream_hash(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(4*1024*1024), b""):
        h.update(b)
    return h.hexdigest().upper()


def names(z):
    seen=set()
    for item in z.infolist():
        n=item.filename
        p=PurePosixPath(n)
        require(not p.is_absolute() and n==p.as_posix() and ".." not in p.parts
                and re.fullmatch(r"[A-Za-z0-9_.+/-]+",n) is not None and n.lower() not in seen,
                "Unsafe/duplicate archive member: "+n)
        require(item.flag_bits&1==0 and (item.external_attr>>16)&0o170000!=0o120000,"Encrypted member or symlink: "+n)
        seen.add(n.lower())


class Slice(io.RawIOBase):
    def __init__(self,path,item):
        self.f=open(path,"rb")
        self.f.seek(item.header_offset)
        header=self.f.read(30)
        require(header[:4]==b"PK\x03\x04" and item.compress_type==ZIP_STORED and item.compress_size==item.file_size,
                "Transport member is not STORE")
        a,b=struct.unpack_from("<HH",header,26)
        self.first=item.header_offset+30+a+b
        self.size=item.file_size
        self.pos=0

    def read(self,n=-1):
        limit=max(0,self.size-self.pos)
        n=limit if n is None or n<0 else min(n,limit)
        self.f.seek(self.first+self.pos)
        b=self.f.read(n)
        self.pos+=len(b)
        return b

    def seek(self,n,whence=0):
        n=n+(0 if whence==0 else self.pos if whence==1 else self.size)
        require(n>=0,"Negative slice seek")
        self.pos=n
        return n

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def close(self):
        self.f.close()
        super().close()
