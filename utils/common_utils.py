from string import punctuation


class Logging:
    def __init__(self, file_name):
        self.file_name = file_name

    def logging(self, info):
        with open(self.file_name, 'a+') as f:
            print(info, file=f)
        print(info)


def get_stop_words():
    stop_words = []#set()
    for i in punctuation:
        stop_words.append('Ġ' + i)
    return stop_words

stop_words = get_stop_words()

