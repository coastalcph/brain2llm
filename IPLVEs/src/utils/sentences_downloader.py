import json
import os
import threading

import fasttext as ft
import requests

ft.FastText.eprint = lambda x:None

class Crawl_sentences(threading.Thread):
    def __init__(self, threadID, name, wordlist, index, download_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.index = index
        self.name = name
        self.download_path = download_path
        self.wordlist = wordlist
    def run(self):
        print(f"Begin: {self.name}")
        create_sentences_file(self.wordlist, self.index, self.threadID, self.download_path)
        print(f"End: {self.name}")

def is_english(text, model):
    return model.predict(text)[0][0] == '__label__en'

def check_symbols(s):
    arr = []
    SYMBOLS = {'}': '{', ']': '[', ')': '(', '>': '<', '"':'"'}
    SYMBOLS_L = SYMBOLS.values()
    for c in s:
        if c in SYMBOLS_L:
            # push symbol left to list
            arr.append(c)
        # pop out symbol,
        elif arr and c in SYMBOLS.keys() and arr[-1] == SYMBOLS[c]:
            arr.pop()
        else:
            pass
    if arr:
        return False
    else:
        return True

def crawl_sentences(word, model):
    """
    Download sentences related to the word from wiki
    
    Args:
      word: the word you want to crawl
      model: the model you want to use to crawl the sentences.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'}

    sentence_list = []
    # ft_model = ft.load_model("./pretrained/lid.176.bin")

    try:
        r = requests.get(f'https://api.rhymezone.com/words?max=501&nonorm=1&k=rz_wke&rel_wke={word}', headers=headers).text

        if len(r) > 2:
            try:
                start = r.index('[{')
                end = r.index('}]') + 2
                data = json.loads(r[start:end])
            except Exception:
                print(f"loaded error: {word}")
            for item in data:
                item_info = item['word'].replace('<b>', '').replace('</b>', '').split(':', 3)
                # print(item_info)
                # Only select the wiki source
                item_sentence = item_info[-1]
                # if (item_info[0] == 'd' or item_info[0] == 'b' or item_info[0] == 'q')
                if (item_info[0] == 'd') \
                        and len(item_sentence.split(' ')) < 16 and '...' not in item_sentence\
                        and check_symbols(item_sentence) and item_sentence.count('"')%2 == 0 \
                        and item_sentence.isascii() and is_english(item_sentence, model): 
                    source = f'{word}: {item_sentence}'
                    sentence_list.append(source)

    except Exception:
        print(f"request error:{word}")
    return sentence_list

def create_sentences_file(wordslist, index, part, download_path):
    """
    Create the file saving part of sentences lists. Using it to active the multi-processing to speed up.
    
    Args:
      wordslist: a list of words
      index: the index of the word in the wordslist
      part: the part of the corpus you want to download (1-5)
      download_path: the path to the directory where the downloaded files are stored
    """
    with open(wordslist, 'r') as words_read:
        words = words_read.readlines()[index:index+2400]
        examples = []
        ft_model = ft.load_model("./pretrained/lid.176.bin")
        for word_n in words:
            word = word_n.strip('\n').replace(' ','_')
            # print(word)

            crawled_sentences = crawl_sentences(word=word, model=ft_model)
            if len(crawled_sentences) >= 5:
                examples.append(crawled_sentences[:15])
            # else:
            #     words.remove(word_n)
        words_read.close()
        if examples:
            if not os.path.exists(download_path):
                os.mkdir(download_path)
            with open(f'{download_path}/sentences_{part}.txt', 'w') as sentences_write:
                for sentences in examples:
                    for sentence in sentences:
                        sentences_write.write(sentence + '\n')
                sentences_write.close()

def sum_files(root_path, sentences_path, args):
    """
    Combine all the downloaded sentences files.
    
    Args:
      root_path: the path to the root directory of the project
      sentences_path: the path to the sentences file
      args: a list of arguments that you can pass to the script.
    """
    files = os.listdir(root_path)
    words_repeated = []
    sentences = []
    for file in files:
        with open(f'{root_path}/{file}', 'r') as file_reader:
            file_sentences = file_reader.readlines()
            sentences += file_sentences
            for sentence in file_sentences:
                word = sentence.split(":", 1)[0]
                # print(word)
                words_repeated.append(word)

    words = list(dict.fromkeys(words_repeated))
    sentences_no_repeat = list(dict.fromkeys(sentences))

    with open(args.data.get("wordlist_path", "./data/wordlist_downloaded.txt"),'w+') as word_w:
        for word in words:
            word_w.write(f'{word}\n')
        word_w.close()

    with open(sentences_path,'w') as sent_w:
        for sent in sentences_no_repeat:
            sent_w.write(sent.strip('\n')+'\n')
        sent_w.close()


def sentences_download(args):
    """
    It downloads the sentences from the internet through multi-processing.
    
    Args:
      args: a list of arguments passed to the function.
    """
    wordlist = args.data.wordlist_path
    sentences_path = args.data.sentences_path
    download_path = args.data.download_path
    if not os.path.exists(sentences_path):
        print('Begin downloading sentences.')
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        threads = []
        for i in range(60):
            thread_index = Crawl_sentences(i, f"Thread-{i}", wordlist=wordlist, index=i*2400, download_path=download_path)
            threads.append(thread_index)

        for thre in threads:
            thre.start()

        for thre in threads:
            thre.join()
        sum_files(download_path, sentences_path, args)
    print('Download sentences successfully.')