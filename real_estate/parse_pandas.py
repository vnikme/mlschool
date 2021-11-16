import sys
import pandas as pd


def main():
    input_path = sys.argv[1]
    df = pd.read_csv(input_path)
    print(df.describe().iloc[0])
    flt = (df['CRIM'] > 1.0)
    print(flt.describe())
    crim = df[flt]
    print(crim.describe())
    print(df.describe())
    y = df['MEDV']
    x = df[df.columns[:-1]]
    print(y.describe())
    print(x.describe())


if __name__ == '__main__':
    main()

