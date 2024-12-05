import hashlib


def string_sha256(s: str) -> str:
    hash_object = hashlib.new("sha256")
    hash_object.update(s.encode('utf-8'))
    hash_value = hash_object.hexdigest()
    return hash_value
