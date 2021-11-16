import sys


def main():
    input_path = sys.argv[1]
    result, samples = None, 0
    for line in open(input_path, 'rt'):
        if line and line[-1] == '\n':
            line = line[:-1]
        if not line:
            continue
        columns = line.split(',')
        try:
            columns = list(map(float, columns))
        except:
            print(columns)
            continue
        if result is None:
            result = [0.0 for _ in range(len(columns))]
        for i in range(len(columns)):
            result[i] += columns[i]
        samples += 1
    for i in range(len(result)):
        result[i] /= samples
    print(result)


if __name__ == '__main__':
    main()

