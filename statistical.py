from operator import pos
from re import U
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

plt.rcParams["figure.figsize"] = [10, 4]


from data_extraction import is_emoji, is_hashtag, is_tag, is_url, consecutive 

def loading_sents():
    train_sents = np.load('./npy_data/train_sents.npy', allow_pickle=True)
    test_sents = np.load('./npy_data/test_sents.npy', allow_pickle=True)
    dev_sents = np.load('./npy_data/dev_sents.npy', allow_pickle=True)
    return train_sents, test_sents, dev_sents
    
def sentence_stats(sents):
    number_sentences = len(sents)
    len_sents = []
    max = 0
    min = sys.maxsize
    word_sum = 0
    for sent in sents: 
        if len(sent) > max:
            max = len(sent)
        if len(sent) < min:
            min = len(sent)
        word_sum += len(sent)
        len_sents.append(len(sent))
    mean = word_sum/number_sentences
    return len_sents, [mean, min, max, word_sum]

def plot_histo_sent(len_sents):
    plt.hist(len_sents, density=False, bins=30)
    plt.ylabel('Occurences (No. of sentences)')
    plt.xlabel('Sentence Length')
    plt.show()


def get_freq(sents):
    freqs = []
    for i in range(1,3):
        flat_list = [item for sublist in sents for item in sublist]
        (unique, counts) = np.unique([x[i] for x in flat_list], return_counts=True)
        freq = [np.asarray((unique, counts/len(flat_list), counts)).T]
        freqs.append(freq)
    return freqs

def statistical_analysis(sents):
    len_sents, sents_stats = sentence_stats(sents)
    # plotting the histogram for the sentences' length
    plot_histo_sent(len_sents=len_sents)
    # pos 0 is pos_tag, pos 1 is bios
    freqs = get_freq(sents=sents)

    flat_list = [item for sublist in sents for item in sublist]
    e_count = np.sum([is_emoji(x[0]) for x in flat_list])
    h_count = np.sum([is_hashtag(x[0]) for x in flat_list])
    t_count = np.sum([is_tag(x[0]) for x in flat_list])
    url_count = np.sum([is_url(x[0]) for x in flat_list])
    consecutive_count = np.sum([consecutive(x[0]) for x in flat_list])
    counts = [e_count, h_count, t_count, url_count, consecutive_count]
    return sents_stats + counts + freqs
    
def filter_bios(bios):
    filtered = list(filter(lambda x: x[0].startswith('B'), bios[0]))
    return list(filtered)


# used to find the max value of a list of lists
def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])

def plot_bios_freq(bios_train, bios_test, bios_dev):
    train = np.array([row[1] for row in bios_train], dtype=float)
    train = train / np.sum(train)

    test = np.array([row[1] for row in bios_test], dtype=float)
    test = test / np.sum(test)

    dev = np.array([row[1] for row in bios_dev], dtype=float)
    dev = dev / np.sum(dev)


    index = [row[0][2:] for row in bios_dev]
    df = pd.DataFrame({'train': train,
                    'test': test,
                    'dev': dev}, index=index)

    ax = df.plot.bar(rot=0)
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_xlabel('NER tags', fontsize=12, fontweight='bold')
    ax.set_title('Frequency of each NER tag', fontsize=14, fontweight='bold' )


    # ax.set_ylim([0, 1.2])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():0.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 20), textcoords='offset points',
                    rotation='vertical')
    ax.margins(0.21)

    plt.show()

if __name__ == "__main__":
    train_sents, test_sents, dev_sents = loading_sents()
    stats_train = statistical_analysis(train_sents)
    bios_train = filter_bios(stats_train[len(stats_train)-1])
    pos_train = stats_train[len(stats_train)-2]

    stats = statistical_analysis(test_sents)
    bios_test = filter_bios(stats[len(stats)-1])
    pos_test = stats[len(stats)-2]

    stats = statistical_analysis(dev_sents)
    bios_dev = filter_bios(stats[len(stats)-1])
    pos_dev = stats[len(stats)-2]

    bios_train = list(map(list, bios_train))
    bios_test = list(map(list, bios_test))
    bios_dev = list(map(list, bios_dev))

    # print(list(sorted(pos_train[0], key=lambda x: float(x[1]), reverse=True))[:5], '\n')
    # print(list(sorted(pos_test[0], key=lambda x: float(x[1]), reverse=True))[:5], '\n')
    # print(list(sorted(pos_dev[0], key=lambda x: float(x[1]), reverse=True))[:5], '\n')

    plot_bios_freq(bios_train, bios_test, bios_dev)
