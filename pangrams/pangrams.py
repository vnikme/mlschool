import time


alphabet = 'abcdefghijklmnopqrstuvwxyz'


def run_and_measure(func, count, text):
    print(func(text))
    start = time.time()
    for _ in range(count):
        func(text)
    finish = time.time()
    print(finish - start)
    print('')


def check_for_every_digit(text):
    for sym in alphabet:
        if sym not in text.lower():
            print(sym, text)
            exit(0)
            return False
    return True


def check_for_every_digit_without_lower(text):
    text = text.lower()
    for sym in alphabet:
        if sym not in text:
            return False
    return True


def check_via_flags(text):
    text = text.lower()
    flags = [False for _ in range(26)]
    for sym in text:
        code = ord(sym) - ord('a')
        if 0 <= code < 26:
            flags[code] = True
    return all(flags)


def check_via_set(text):
    used = set()
    for sym in text.lower():
        if 'a' <= sym <= 'z':
            used.add(sym)
    return len(used) == 26


def run_all(count, text):
    print(len(text))
    run_and_measure(check_for_every_digit, count, text)
    run_and_measure(check_for_every_digit_without_lower, count, text)
    run_and_measure(check_via_flags, count, text)
    run_and_measure(check_via_set, count, text)


def main():
    run_all(10000, 'Jackdaws love my big sphinx of quartz')
    text = ''
    for i in range(len(alphabet)):
        for _ in range(1000000 if i < 2 else 10):
            text += alphabet[i]
    run_all(1, text)


if __name__ == '__main__':
    main()

