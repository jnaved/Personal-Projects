def search(text: str, pattern: str):
    n = len(text)
    m = len(pattern)
    res = []

    for i in range(n-m+1):
        k = i
        flag = True
        for j in range(m):
            if text[k] == pattern[j]:
                j+=1
                k+=1
            else:
                flag = False
                break
        
        if flag:
            res.append(i)
    
    return res


def main(text: str, pattern: str):
    search(text=text, pattern=pattern)


if __name__ == "__main__":
    text = "aabaacaadaabaaba"
    pattern = "aaba"

    res = search(text=text, pattern=pattern)
    print(res)