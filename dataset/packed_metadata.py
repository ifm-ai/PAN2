import io


class FileSegment(io.RawIOBase):
    def __init__(self, file_path, offset, size):
        self.f = open(file_path, 'rb')
        self.f.seek(offset)
        self.start = offset
        self.size = size
        self.pos = 0

    def read(self, n=-1):
        if n == -1 or n > self.size - self.pos:
            n = self.size - self.pos
        if n <= 0:
            return b""
        data = self.f.read(n)
        self.pos += len(data)
        return data

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = max(0, min(offset, self.size))
        elif whence == io.SEEK_CUR:
            self.pos = max(0, min(self.pos + offset, self.size))
        elif whence == io.SEEK_END:
            self.pos = max(0, min(self.size + offset, self.size))
        self.f.seek(self.start + self.pos)
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

