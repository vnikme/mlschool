import json
import urllib.request
import urllib.parse


def sum(a, b):
    url = 'http://localhost:8197/sum'
    data = {'a': a, 'b': b}
    request = urllib.request.Request(url, method='POST', data=json.dumps(data).encode())
    response = urllib.request.urlopen(request)
    #print(response.reason, response.status)
    answer = response.read()
    answer = json.loads(answer)
    print(answer)
    print('')


def main():
    sum(2, 2)


if __name__ == '__main__':
    main()

