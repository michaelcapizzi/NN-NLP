
def accuracy(correct, total):
    if correct == 0:
        return float(0)
    else:
        return float(correct) / float(total)


def precision(tp, fp):
    if tp == 0:
        return float(0)
    else:
        return float(tp) / float(tp + fp)


def recall(tp, fn):
    if tp == 0:
        return 0
    else:
        return float(tp) / float(tp + fn)


def f1(p, r):
    if p == 0 or r == 0:
        return float(0)
    else:
        return float(2 * p * r) / float(p + r)

