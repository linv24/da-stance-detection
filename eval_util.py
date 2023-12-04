import bert_score
from bert_score import BERTScorer
import matplotlib.pyplot as plt


def bertscore_preprocess(src: str, augs: list[str]):
    '''
    Given the original (src) data sample and a list of augmentations created from the
    original, return two lists suitable for input to BERTScore: a cands list,
    and a refs list that just contains the original data sample duplicated
    for each augmentation/candidate
    '''
    cands = augs
    refs = [src] * len(augs)

    return cands, refs

def get_bertscore(scorer, src, augs):
    '''
    Given a BERTScore instance, a source tweet, and its generated augmentations,
    return the BERTScore F1 score of each augmentation with the source.
    BERTScore takes a while to initialize, so it's best to create and reuse a
    single instance.

    BERTScore instance can be created as follows:
    >>> scorer = BERTScorer(lang='en', rescale_with_baseline=True)
    '''
    cands, refs = bertscore_preprocess(src, augs)
    P, R, F1 = scorer.score(cands, refs)
    return F1

def get_f_avg(P_favor, R_favor, P_against, R_against):
    '''
    Given precision and recall scores for "favor" and "against" stances, return
    an F1 score averaged between the two stances.
    '''
    F_favor = (2 * P_favor * R_favor) / (P_favor + R_favor)
    F_against = (2 * P_against * R_against) / (P_against + R_against)
    F_avg = (F_favor + F_against) / 2

    return F_avg


if __name__ == '__main__':
    scorer = BERTScorer(lang='en', rescale_with_baseline=True)

    cands_str = '''28-year-old chef found dead in San Francisco mall
    A 28-year-old chef who recently moved to San Francisco was found dead in the staircase of a local shopping center.
    The victim's brother said he cannot imagine anyone who would want to harm him,"Finally, it went uphill again at him."
    The corpse, found Wednesday morning in the Westfield Mall, was identified as the 28-year-old Frank Galicia from San Francisco, the Justice Department said in San Francisco.
    The San Francisco Police Department said the death was classified as murder and the investigation is on the running.
    The victim's brother, Louis Galicia, told the ABS broadcaster KGO in San Francisco that Frank, who formerly worked as a cook in Boston, had his dream job as a cook at the Sons & Daughters restaurant in San Francisco six months ago.
    A spokesman for the Sons & Daughters said they were "shocked and destroyed on the ground" over his death.
    "We are a small team that works like a close family and we are going to miss him painfully," said the spokesman.
    Our thoughts and condolences are in this difficult time at Franks's family and friends.
    Louis Galicia admitted that Frank initially lived in hostels, but that "things for him finally went uphill."'''

    refs_str = '''28-Year-Old Chef Found Dead at San Francisco Mall
    A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.
    But the victim's brother says he can't think of anyone who would want to hurt him, saying, "Things were finally going well for him."
    The body found at the Westfield Mall Wednesday morning was identified as 28-year-old San Francisco resident Frank Galicia, the San Francisco Medical Examiner's Office said.
    The San Francisco Police Department said the death was ruled a homicide and an investigation is ongoing.
    The victim's brother, Louis Galicia, told ABC station KGO in San Francisco that Frank, previously a line cook in Boston, had landed his dream job as line chef at San Francisco's Sons & Daughters restaurant six months ago.
    A spokesperson for Sons & Daughters said they were "shocked and devastated" by his death.
    "We are a small team that operates like a close knit family and he will be dearly missed," the spokesperson said.
    Our thoughts and condolences are with Frank's family and friends at this difficult time.
    Louis Galicia said Frank initially stayed in hostels, but recently, "Things were finally going well for him."'''

    # cands/refs must be lists of strings, with each cand in cands list aligned with corresponding ref in refs list
    # refs may be a string, or a nested list of strings from which the best ref will be used for scoring
    cands = [l.strip() for l in cands_str.split('\n')]
    refs = [l.strip() for l in refs_str.split('\n')]

    # must specify either lang or model
    scorer = BERTScorer(lang='en', rescale_with_baseline=True)

    P, R, F1 = scorer.score(cands, refs, verbose=True)

    print(F1)

    plt.hist(F1, bins=20)
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.show()