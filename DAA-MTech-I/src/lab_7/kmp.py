def construct_lps(pattern: str, lps: list):
    m = len(pattern)
    pattern_length = 0

    lps[0] = 0
    i=1
    while(i<m):
        if pattern[i] == pattern[pattern_length]:
            pattern_length += 1
            lps[i] = pattern_length
            i+=1
        else:
            if pattern_length != 0:
                pattern_length = lps[pattern_length-1]
            else:
                lps[i] = 0
                i+=1



def search(text: str, pattern: str):
    n = len(text)
    m = len(pattern)

    # lps -> longest proper prefix which is also a suffix
    lps = [0]*m
    res = []

    construct_lps(pattern=pattern, lps=lps)
    i, j = 0, 0

    while(i<n):
        if text[i] == pattern[j]:
            i+=1
            j+=1

            if j==m:
                res.append(i-j)
                j = lps[j-1]
        else:
            if j!=0:
                j = lps[j-1]
            else:
                i+=1
    return res


def main(text: str, pattern: str):
    search(text=text, pattern=pattern)


if __name__ == "__main__":
    text = "aabaacaadaabaaba"
    pattern = "aaba"
    res = search(text=text, pattern=pattern)
    print(res)