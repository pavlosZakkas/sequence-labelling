import re  # regular expression module
import emoji

EMOJI_STRINGS = [
    "(.V.)", "O:-)", "X-(", "~:0", ":-D", "(*v*)", ":-#", "</3", "=^.^=", "*<:o)",
    "O.o", "B-)", ":'(", ":_(", "\:D/", "*-*", ":o3", "#-o", ":*)", "//_^", ">:)",
    "<><", ":(", ":-(", "=P", ":-P", "8-)", "$_$", ":->", ":-)", ":)", "=)", "<3",
    ":-|", "X-p", "xP", ":-*", ":-)*", "(-}{-)", "xD", "XD", "=D", ")-:", "(-:", "=/",
    ":-)(-:", "~,~", ":-B", "^_^", "<l:0", ":-/", "=8)", ":O", ":P", ":o", ":-E"
]


def is_tag(word):
    return word[0] == '@'


def is_hashtag(word):
    return word[0] == '#'


def is_url(word):
    return word.startswith('http')


def consecutive(string):
    if re.search(r'([a-zA-Z])\1\1', string):
        return True
    else:
        return False


def is_emoji(word):
    def contains_emoji_string(word):
        return any(emoji in word for emoji in EMOJI_STRINGS)

    def contains_unicoded_emoji(word):
        return any(char in emoji.UNICODE_EMOJI['en'] for char in word)

    return True if contains_emoji_string(word) or contains_unicoded_emoji(word) else False


def get_extra_features(word, feature_prefix, extra_features):
    # adding potentially extra features
    features = {}

    features.update({f'{feature_prefix}word.is_emoji()': is_emoji(word)}
                    ) if 'emoji' in extra_features else None
    features.update({f'{feature_prefix}word.is_tag()': is_tag(word)}
                    ) if 'tag' in extra_features else None
    features.update({f'{feature_prefix}word.is_hashtag()': is_hashtag(word)}
                    ) if 'hashtag' in extra_features else None
    features.update({f'{feature_prefix}word.consecutive()': consecutive(word)}
                    ) if 'consecutive' in extra_features else None
    features.update({f'{feature_prefix}word.url()': is_url(word)}
                    ) if 'url' in extra_features else None

    return features


def get_pre_post_word_features(sent, word_position, search_depth, extra_features):
    features = {}
    if word_position > search_depth-1:
        previous_word = sent[word_position-1][0]
        previous_postag = sent[word_position-1][1]

        features.update(get_extra_features(
            previous_word, f'-{search_depth}:', extra_features))
        features.update({
            f'-{search_depth}:word.lower()': previous_word.lower(),
            f'-{search_depth}:word.istitle()': previous_word.istitle(),
            f'-{search_depth}:word.isupper()': previous_word.isupper(),
            f'-{search_depth}:postag': previous_postag,
            f'-{search_depth}:postag[:2]': previous_postag[:2],
        })
    else:
        features['BOS'] = True

    if word_position < len(sent) - search_depth - 1:
        next_word = sent[word_position+1][0]
        next_postag = sent[word_position+1][1]
        features.update(get_extra_features(
            next_word, f'+{search_depth}:', extra_features))
        features.update({
            f'+{search_depth}:word.lower()': next_word.lower(),
            f'+{search_depth}:word.istitle()': next_word.istitle(),
            f'+{search_depth}:word.isupper()': next_word.isupper(),
            f'+{search_depth}:postag': next_postag,
            f'+{search_depth}:postag[:2]': next_postag[:2],
        })
    else:
        features['EOS'] = True

    return features


def word2features(sent, i, extra_features=[], search_depth=1):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    features.update(get_extra_features(word, '', extra_features))
    for depth in range(1, search_depth + 1):
        pre_post_word_features = get_pre_post_word_features(
            sent, i, depth, extra_features)
        features.update(pre_post_word_features)

    return features


def sent2features(sent, extra_feat=[], search_depth=1):
    return [word2features(sent, i, extra_features=extra_feat, search_depth=search_depth)
            for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]
