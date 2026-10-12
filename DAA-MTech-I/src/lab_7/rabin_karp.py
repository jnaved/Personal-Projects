Q = 101


def search(text: str, pattern: str):
    n = len(text)
    m = len(pattern)

    res = []
    h = pow(10, m-1, Q)

    p_hash, t_hash = 0, 0

    # initial hash
    for i in range(m):
        p_hash = (10 * p_hash + ord(pattern[i])) % Q
        t_hash = (10 * t_hash + ord(text[i])) % Q

    for i in range(n-m+1):
        if p_hash == t_hash:
            if text[i: i+m] == pattern:
                res.append(i)

        if i < n-m:
            t_hash = (10 * (t_hash - ord(text[i]) * h) + ord(text[i+m])) % Q
            if t_hash < 0:
                t_hash += Q

    return res


def main(text: str, pattern: str):
    search(text=text, pattern=pattern)


if __name__ == "__main__":
    text = "aabaacaadaabaaba"
    pattern = "aaba"
    res = search(text=text, pattern=pattern)
    print(res)