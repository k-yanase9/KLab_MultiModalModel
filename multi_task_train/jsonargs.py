import json
import sys


def main():
    # 引数からjson文字列を取得する
    json_str = sys.argv[1]

    # json文字列をパースして辞書型に変換する
    json_dict = json.loads(json_str)

    # 辞書型を表示する
    print(json_dict)


if __name__ == '__main__':
    main()
