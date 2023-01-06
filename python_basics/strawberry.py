def read_data():
    n, m = list(map(int, input().split()))
    a = []
    for _ in range(n):
        a.append(input())
    return a
 

def solve(a):
    n, m = len(a), len(a[0])
    rows_with_s = [False for _ in range(n)]
    cols_with_s = [False for _ in range(m)]
    for i in range(n):
        for j in range(m):
            if a[i][j] == 'S':
                rows_with_s[i] = True
                cols_with_s[j] = True
    number_of_rows_without_s, number_of_cols_without_s = 0, 0
    for i in range(n):
        if not rows_with_s[i]:
            number_of_rows_without_s += 1
    for j in range(m):
        if not cols_with_s[j]:
            number_of_cols_without_s += 1
    return number_of_cols_without_s * n + number_of_rows_without_s * (m - number_of_cols_without_s)


def output(result):
    print(result)


def main():
    output(solve(read_data()))


if __name__ == '__main__':
    main()

