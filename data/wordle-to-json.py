import json
import datetime as dt

def parse_words_file(txt_path, json_path):
    result = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n': continue
            line = line.split('-')
            wordle = line[0][-6:-1]
            date = line[1][1:-1]
            if date[1] == ' ': date = '0' + date
            date = dt.datetime.strptime(date, "%d %B %Y").strftime("%Y-%m-%d")
            print(wordle, date)

            # add to dictionary:
            result[date] = wordle

    # add to json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    parse_words_file('words.txt', 'temp_words.json')