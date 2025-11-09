from pathlib import Path
import hashlib
import time

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def main():
    plaintexts = [
        "pass1",
        "password",
        "letmein123",
        "qwerty2020",
        "P@ssw0rd!",
        "abc123",
        "admin",
    ]

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = Path("passwd_cracker/hashcat/generated") / ts
    base.mkdir(parents=True, exist_ok=True)

    md5_file = base / "hashes_md5.txt"
    sha1_file = base / "hashes_sha1.txt"
    sha256_file = base / "hashes_sha256.txt"
    plain_file = base / "plaintexts.txt"

    with md5_file.open("w") as f_md5, sha1_file.open("w") as f_sha1, sha256_file.open("w") as f_sha256, plain_file.open("w") as f_plain:
        for p in plaintexts:
            f_plain.write(p + "\n")
            f_md5.write(md5_hex(p) + "\n")
            f_sha1.write(sha1_hex(p) + "\n")
            f_sha256.write(sha256_hex(p) + "\n")

    print("Wrote hashes to:", md5_file, sha1_file, sha256_file)
    print("Plaintexts saved to:", plain_file)

if __name__ == "__main__":
    main()

