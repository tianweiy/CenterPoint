"""\
This module offser helpers for OSS operation.

Basic Use
----------
Create an :class:`OSSPath` object::

    >>> p = OSSPath('s3://mybucket/myprefix/mykey.bin')
    OSSPath('s3://mybucket/myprefix/mykey.bin')
    >>> OSSPath() / "mybucket" / "myprefix" / "mykey.bin"
    OSSPath('s3://mybucket/myprefix/mykey.bin')


Querying object properies::

    >>> p.exists()
    True
    >>> p.is_dir()
    False
    >>> p.is_file()
    True
    >>> p.get_size()
    256

Access path properties::

    >>> p.bucket
    "mybucket"
    >>> p.key
    "myprefix/mykey.bin"
    >>> p.name
    "mykey.bin"
    >>> p.stem
    "mykey"
    >> p.suffix
    ".bin"
    >> p.suffixes
    [".bin"]
    >>> p.parent
    OSSPath('s3://mybucket/myprefix')
    >>> p.root
    OSSPath('s3://mybucket')

Uploading content to an object::

    >>> p.put(b"some bytes\n")
    True

Uploading file to an object::

    >>> p.put(open('/path/some/image.jpg', 'rb'))

Reading an object::

    >>> f = p.download()
    >>> f.read()
    b"some bytes"
    >>> p.download(encoding='utf-8')
    >>> f.read()
    "some bytes"

Deleting an object::

    >>> p.delete()
    True

Path manipulations::

    >>> p = OSSPath('s3://mybucket/myprefix/mykey.bin')
    >>> p.with_name('mykey2.bin')
    OSSPath("s3://mybucket/myprefix/mykey2.bin")
    >>> p.with_suffix('.txt')
    OSSPath("s3://mybucket/myprefix/mykey.txt")
    >>> p.with_bucket('some_bucket')
    OSSPath("s3://some_bucket/myprefix/mykey.txt")

    >>> q = p.parent
    >>> q
    OSSPath('s3://mybucket/myprefix')
    >>> q / "subfile.txt"
    OSSPath("s3://mybucket/myprefix/subfile.txt")
    >>> q / "subdir" / "subfile.txt"
    OSSPath("s3://mybucket/myprefix/subdir/subfile.txt")
    >>> q.joinpath("a", "b", "c")
    OSSPath('s3://mybucket/myprefix/a/b/c')

Directory-level operations::

    >>> list(q.list_all())  # list all subfiles in all levels
    >>> list(q.iter_dir())  # list subdirs and subfiles in one-level
    >>> for root, dirs, files in q.walk(): print(files)  # recursively walk through directory
    >>> q.rmtree()  # remove all subkeys of p


"""
import os
import io
import codecs
from typing import Tuple, Iterable, Optional, List
from pathlib import PosixPath
from urllib.parse import urlparse, urlunparse
import re
import socket
import boto3
from botocore.errorfactory import ClientError


def get_site():
    m = re.search(r"([^.]+)\.brainpp\.cn$", socket.getfqdn())
    if m:
        return m.group(1)


OSS_ENDPOINT = os.getenv(
    "OSS_ENDPOINT", default="http://oss.{}.brainpp.cn".format(get_site()),
)


class OSSPath:

    __slots__ = ("_client", "bucket", "_key_parts")

    def __new__(cls, s3url: Optional[str] = None, endpoint_url=OSS_ENDPOINT):
        _client = boto3.client("s3", endpoint_url=endpoint_url)
        bucket, parts = cls._parse_s3url(s3url)
        return cls._create(_client, bucket, parts)

    @classmethod
    def _parse_s3url(cls, s3url: Optional[str] = None):
        if s3url is None:
            return "", ()

        if not s3url.startswith("s3://"):
            raise ValueError(
                "s3url must be formated as 's3://<bucket_name>/path/to/object'"
            )

        r = urlparse(s3url)
        assert r.scheme == "s3"

        key = r.path.lstrip("/")  # remove the leading /

        parts = PosixPath(key).parts
        return r.netloc, parts

    @classmethod
    def _create(cls, client, bucket: str, key_parts: Tuple[str]):
        assert isinstance(key_parts, tuple)
        self = object.__new__(cls)
        self._client = client
        self.bucket = bucket
        self._key_parts = key_parts
        return self

    @property
    def key(self) -> str:
        return "/".join(self._key_parts)

    @property
    def parent(self):
        """The logical parent of the path."""

        if not len(self._key_parts):
            return self

        return self._create(self._client, self.bucket, self._key_parts[:-1])

    @property
    def root(self):
        return self._create(self._client, self.bucket, key_parts=())

    @property
    def name(self):
        if len(self._key_parts) < 1:
            return ""
        return self._key_parts[-1]

    @property
    def suffix(self):
        """The final component's last suffix, if any."""
        name = self.name
        i = name.rfind(".")
        if 0 < i < len(name) - 1:
            return name[i:]
        else:
            return ""

    @property
    def suffixes(self):
        """A list of the final component's suffixes, if any."""
        name = self.name
        if name.endswith("."):
            return []
        name = name.lstrip(".")
        return ["." + suffix for suffix in name.split(".")[1:]]

    @property
    def stem(self):
        """The final path component, minus its last suffix."""
        name = self.name
        i = name.rfind(".")
        if 0 < i < len(name) - 1:
            return name[:i]
        else:
            return name

    @property
    def parts(self):
        """An object providing sequence-like access to the
        components in the filesystem path."""

        return self._key_parts

    def __str__(self) -> str:
        return "s3://{}/{}".format(self.bucket, self.key)

    def __eq__(self, other):
        if not isinstance(other, OSSPath):
            return False
        return self.bucket == other.bucket and self.key == other.key

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self))

    def __lt__(self, other):
        if not isinstance(other, OSSPath):
            raise NotImplementedError()
        return str(self) < str(other)

    def __le__(self, other):
        if not isinstance(other, OSSPath):
            raise NotImplementedError()
        return str(self) <= str(other)

    def __gt__(self, other):
        if not isinstance(other, OSSPath):
            raise NotImplementedError()
        return str(self) > str(other)

    def __ge__(self, other):
        if not isinstance(other, OSSPath):
            raise NotImplementedError()
        return str(self) >= str(other)

    def with_name(self, name):
        """Return a new path with the file name changed."""
        if not self.name:
            raise ValueError("%r has an empty name" % (self,))

        r = urlparse(name)
        if not (r.scheme == "" and r.netloc == "" or "/" in name):
            raise ValueError("invalid name %r" % (name))

        return self._create(self._client, self.bucket, self._key_parts[:-1] + (name,))

    def with_suffix(self, suffix):
        """Return a new path with the file suffix changed.  If the path
        has no suffix, add given suffix.  If the given suffix is an empty
        string, remove the suffix from the path.
        """
        if "/" in suffix:
            raise ValueError("Invalid suffix %r" % (suffix,))
        if suffix and not suffix.startswith(".") or suffix == ".":
            raise ValueError("Invalid suffix %r" % (suffix))
        name = self.name
        if not name:
            raise ValueError("%r has an empty name" % (self,))
        old_suffix = self.suffix
        if not old_suffix:
            name = name + suffix
        else:
            name = name[: -len(old_suffix)] + suffix
        return self._create(self._client, self.bucket, self._key_parts[:-1] + (name,))

    def with_bucket(self, bucket):
        if not isinstance(bucket, str):
            raise ValueError("bucket be string")

        bucket = bucket.strip("/")
        if not bucket:
            raise ValueError("bucket must not be empty")
        if "/" in bucket:
            raise ValueError("bucket_name must not contain '/'")
        return self._create(self._client, bucket, self._key_parts)

    def _make_child(self, args: Iterable[str]):

        if not self.bucket:
            bucket, *rest_args = args
            bucket = bucket.lstrip("/")
            bucket, *rest_parts = PosixPath(bucket).parts
            return self.with_bucket(bucket)._make_child(rest_parts + rest_args)

        parts = [p for p in self._key_parts]
        for item in args:
            if not isinstance(item, str):
                raise ValueError("child must be string")
            item = item.lstrip("/")  # remove leading '/'
            if not item:
                raise ValueError("child must not be empty")
            for p in PosixPath(item).parts:
                parts.append(p)

        return self._create(self._client, self.bucket, tuple(parts))

    def joinpath(self, *args):
        """Combine this path with one or several arguments, and return a
        new path representing either a subpath (if all arguments are relative
        paths) or a totally different path (if one of the arguments is
        anchored).
        """
        return self._make_child(args)

    def __truediv__(self, key):
        return self._make_child((key,))

    def __rtruediv__(self, key):
        raise NotImplemented

    def is_dir(self):
        if not self.bucket:
            return False

        if not self.key:
            # key empty, return whether bucket exists
            try:
                self._client.head_bucket(Bucket=self.bucket)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False

        prefix = self.key
        if prefix[-1] != "/":
            prefix = prefix + "/"
        resp = self._client.list_objects(
            Bucket=self.bucket, Delimiter="/", Prefix=prefix
        )
        return "CommonPrefixes" in resp or "Contents" in resp

    def is_file(self):
        if not self.bucket:
            return False
        if not self.key:
            return False
        try:
            self._client.head_object(Bucket=self.bucket, Key=self.key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False

    def exists(self):
        if not self.bucket:
            return False
        if self.is_dir():
            return True
        elif self.is_file():
            return True
        return False

    def get_size(self):
        if not self.bucket:
            return -1
        if self.is_dir():
            return 0
        if not self.is_file():
            return -1

        key = self.key.lstrip("/")
        return self._client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]

    def list_all(self, batch_size=1000):
        """\
        List all subkeys
        :returns: Iterator[OSSPath]
        """
        if not self.is_dir():
            return

        if batch_size > 1000:
            print(
                "At most 1000 keys can be operated at once. Clipping batch_size to 1000."
            )
            batch_size = 1000

        prefix = self.key
        if prefix[-1] != "/":
            prefix = prefix + "/"

        marker = None
        while True:
            request = dict(
                Bucket=self.bucket, Delimiter="", Prefix=prefix, MaxKeys=batch_size,
            )
            if marker:
                request["Marker"] = marker

            resp = self._client.list_objects(**request)

            for p in resp.get("Contents", []):
                yield self.root / p["Key"]

            if not resp["IsTruncated"]:
                break

            print(
                "More than {} objects are found under {}, you should avoid putting too many small objects!".format(
                    batch_size, self
                )
            )
            marker = resp["NextMarker"]

    def walk(self, topdown=True, recursive=True, batch_size=1000):
        """\
        Generate path tree by walking either top-down or bottom-up just like :func:`os.walk`.
        For each prefix in the tree, it yields a 3-tuple (subtree-root, subdirs, subfiles).

        If optional argument *topdown* is True or not specified, the triple for a directory
        is generated before the triples for any subdirectories. If *topdown* is False,
        the triple for a directory is generated after its subdirectries.

        If *recurisve* is set to False, it only yields the top level subdirectries and subfiles.

        *batch_size* is the maximum keys that OSS returns in one request-response,
        and it cannot be set larger than 1000.
        """
        if not self.is_dir():
            return

        if batch_size > 1000:
            print(
                "At most 1000 keys can be operated at once. Clipping batch_size to 1000."
            )
            batch_size = 1000

        prefix = self.key
        if prefix[-1] != "/":
            prefix = prefix + "/"

        dirs, files = [], []
        marker = None
        while True:
            request = dict(
                Bucket=self.bucket, Delimiter="/", Prefix=prefix, MaxKeys=batch_size,
            )
            if marker:
                request["Marker"] = marker

            resp = self._client.list_objects(**request)

            dirs += [self.root / p["Prefix"] for p in resp.get("CommonPrefixes", [])]

            files += [self.root / p["Key"] for p in resp.get("Contents", [])]

            if not resp["IsTruncated"]:
                break

            print(
                "More than {} objects are found under {}, you should avoid putting too many small objects!".format(
                    batch_size, self
                )
            )
            marker = resp["NextMarker"]

        if topdown:
            yield self, dirs, files

        if recursive:
            for subdir in dirs:
                yield from subdir.walk(
                    recursive=True, topdown=topdown, batch_size=batch_size
                )

        if not topdown:
            yield self, dirs, files

    def iterdir(self, batch_size=1000):
        """
        Iterates over self directory, yields subdirs and subfiles.
        :returns: Iterator[OSSPath]
        """
        for root, dirs, files in self.walk(batch_size=batch_size, recursive=False):
            yield from dirs
            yield from files

    def download(self, encoding=None) -> Optional[io.IOBase]:
        """
        :param encoding: if None, it returns bytes io;
            if an encoding (such as 'utf-8') is specified, it returns text io

        :returns: file-like object which can be read out
        """

        if not self.is_file():
            raise FileNotFoundError("{!r} is not an existing object.".format(self))

        r = self._client.get_object(Bucket=self.bucket, Key=self.key)
        b = r["Body"]
        if encoding is not None:
            b = codecs.getreader(encoding)(b)

        return b

    def put(self, bytes_or_file) -> bool:
        """
        :param bytes_or_file: bytes or file-like object to be uploaded to OSS
        :returns: wheter successfully uploaded
        """
        if not self.bucket or not self.key:
            raise ValueError("Invalid path to put object: {!r}".format(self))
        if self.key.endswith("/"):
            raise ValueError('Object key cannot endswith "/": {}'.format(self.key))

        r = self._client.put_object(
            Body=bytes_or_file, Bucket=self.bucket, Key=self.key,
        )
        return r["ResponseMetadata"]["HTTPStatusCode"] == 200

    def delete(self) -> bool:
        """
        :returns: whether this object is deleted
        """
        if not self.is_file():
            return True
        r = self._client.delete_object(Bucket=self.bucket, Key=self.key)

        return r["ResponseMetadata"]["HTTPStatusCode"] == 204

    def rmtree(self, batch_size=1000) -> List[str]:
        """
        :returns: list of deleted objects
        """
        if not self.is_dir():
            if self.is_file():
                raise ValueError("{!r} is not a directory".format(self))
            return True

        if batch_size > 1000:
            print(
                "At most 1000 keys can be operated at once. Clipping batch_size to 1000."
            )
            batch_size = 1000

        prefix = self.key
        if prefix[-1] != "/":
            prefix = prefix + "/"

        ret = []
        while True:
            lr = self._client.list_objects(
                Bucket=self.bucket, Delimiter="", Prefix=prefix, MaxKeys=batch_size,
            )

            dr = self._client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": i["Key"]} for i in lr.get("Contents", [])]},
            )

            for i in dr["Deleted"]:
                ret.append("s3://{}/{}".format(self.bucket, i["Key"]))

            if not lr["IsTruncated"]:
                break

            print(
                "More than {} objects are found under {}, you should avoid putting too many small objects!".format(
                    batch_size, self
                )
            )

        return ret
