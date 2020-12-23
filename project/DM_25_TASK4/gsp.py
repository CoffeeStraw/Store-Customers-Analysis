# coding: utf-8


# # Preliminaries
import copy
import itertools
from collections import defaultdict
from operator import itemgetter

# #### Our dataset format
# An event is a list of strings.
# A sequence is a list of events.
# A dataset is a list of sequences.
# Thus, a dataset is a list of lists of lists of strings.
#
# E.g.
# dataset =  [
#    [["a"], ["a", "b", "c"], ["a", "c"], ["c"]],
#    [["a"], ["c"], ["b", "c"]],
#    [["a", "b"], ["d"], ["c"], ["b"], ["c"]],
#    [["a"], ["c"], ["b"], ["c"]]
# ]


# # Foundations
# ### Subsequences

#This is a simple recursive method that checks if subsequence is a subSequence of mainSequence


def is_subsequence(seq, sub_seq, seq_time_stamps=[], max_span=None, min_gap=None, max_gap=None):
    """
    Iterative function to check whether sub_seq is a sub-sequence of seq.
    """
    # Temporary number for the starting event in the seq
    i = 0

    # No Time Constraints? Simpler algorithm.
    if not seq_time_stamps:
        # Assign a number to each event
        seq = list(enumerate(seq))

        for sub_event in sub_seq:
            sub_event = set(sub_event)
            for e_i, event in seq[i:]:
                if set(event).issuperset(sub_event):
                    # We found a match between two events
                    i = e_i + 1
                    break
            else: # We iterated over all the events in seq without corrispondence
                return False
        return True

    # Time Constraints: we need more variables.
    # Assign a number to each event-ts couple
    seq = list(enumerate(zip(seq, seq_time_stamps)))

    # Temporary variables for time constraints
    start, curr = None, None

    for sub_event in sub_seq:
        sub_event = set(sub_event)
        for e_i, (event, ts) in seq[i:]:
            if set(event).issuperset(sub_event):
                # We found a match between two events
                i = e_i + 1
                
                if not start:
                    # First match: save time stamps for later
                    start = ts
                    curr = ts
                else:
                    # This is at least the second match found: we can check time contraints
                    diff = ts - curr
                    # Check for min_gap validity
                    if min_gap:
                        if diff < min_gap:
                            return False  # Min_gap violated
                    if max_gap:
                        if diff > max_gap:
                            return False  # Max_gap violated
                    curr = ts
                break
        else: # We iterated over all the events in seq without corrispondence
            return False

    # Last check: max_span validity
    if max_span:
        if curr - start > max_span:
            return False  # Max_span violated
    return True


# ### Size of sequences
def sequenceSize(sequence):
    """
    Computes the size of the sequence (sum of the size of the contained elements)
    """
    return sum(len(i) for i in sequence)


# ### Support of a sequence
def countSupport(dataset, candidateSequence, time_stamps=[], max_span=None, min_gap=None, max_gap=None):
    """
    Computes the support of a sequence in a dataset.
    """
    if not time_stamps:
        return round(sum(1 for seq in dataset if is_subsequence(seq, candidateSequence)) / len(dataset), 2)
    return round(sum(1 for seq, ts in zip(dataset, time_stamps) if is_subsequence(seq, candidateSequence, ts, max_span, min_gap, max_gap)) / len(dataset), 2)


# # AprioriAll
# ### 1 . Candidate Generation

# #### For a single pair:
def generateCandidatesForPair(cand1, cand2):
    """
    Generates one candidate of size k from two candidates of size (k-1) as used in the AprioriAll algorithm
    """
    cand1Clone = copy.deepcopy(cand1)
    cand2Clone = copy.deepcopy(cand2)
    # drop the leftmost item from cand1:
    if (len(cand1[0]) == 1):
        cand1Clone.pop(0)
    else:
        cand1Clone[0] = cand1Clone[0][1:]
    # drop the rightmost item from cand2:
    if (len(cand2[-1]) == 1):
        cand2Clone.pop(-1)
    else:
        cand2Clone[-1] = cand2Clone[-1][:-1]

    # if the result is not the same, then we dont need to join
    if not cand1Clone == cand2Clone:
        return []
    else:
        newCandidate = copy.deepcopy(cand1)
        if (len(cand2[-1]) == 1):
            newCandidate.append(cand2[-1])
        else:
            newCandidate[-1].extend(cand2[-1][-1])
        return newCandidate


# #### For a set of candidates (of the last level):
def generateCandidates(lastLevelCandidates):
    """
    Generates the set of candidates of size k from the set of frequent sequences with size (k-1)
    """
    k = sequenceSize(lastLevelCandidates[0]) + 1
    if (k == 2):
        flatShortCandidates = [item for sublist2 in lastLevelCandidates for sublist1 in sublist2 for item in sublist1]
        result = [[[a, b]] for a in flatShortCandidates for b in flatShortCandidates if b > a]
        result.extend([[[a], [b]] for a in flatShortCandidates for b in flatShortCandidates])
        return result
    else:
        candidates = []
        for i in range(0, len(lastLevelCandidates)):
            for j in range(0, len(lastLevelCandidates)):
                newCand = generateCandidatesForPair(lastLevelCandidates[i], lastLevelCandidates[j])
                if (not newCand == []):
                    candidates.append(newCand)
        candidates.sort()
        return candidates


# ### 2 . Candidate Checking
def generateDirectSubsequences(sequence):
    """
    Computes all direct subsequence for a given sequence.
    A direct subsequence is any sequence that originates from deleting exactly one item from any element in the original sequence.
    """
    result = []
    for i, itemset in enumerate(sequence):
        if (len(itemset) == 1):
            sequenceClone = copy.deepcopy(sequence)
            sequenceClone.pop(i)
            result.append(sequenceClone)
        else:
            for j in range(len(itemset)):
                sequenceClone = copy.deepcopy(sequence)
                sequenceClone[i].pop(j)
                result.append(sequenceClone)
    return result


def generateContiguousSubsequences(sequence):
    """
    Compute the (k-1)-contiguous-subsequences of the sequence.
    """
    result = []
    
    if len(sequence[0]) == 1:
        tmp = copy.deepcopy(sequence)
        tmp.pop(0)
        result.append(tmp)  # Append sub-sequence without first element

    if len(sequence[-1]) == 1:
        tmp = copy.deepcopy(sequence)
        tmp.pop(-1)
        result.append(tmp)  # Append sub-sequence without last element

    # For each event with len > 1, generate sub-sequences removing 1 item from them
    for i in range(len(sequence)):
        if len(sequence[i]) > 1:
            for j in range(len(sequence[i])):
                tmp = copy.deepcopy(sequence)
                tmp[i].pop(j)
                result.append(tmp)
    
    # We are not interested in generating all the contiguous-subsequences,
    # but only in the (k-1) ones, so we stop here
    
    return result



# ### Put it all together:
def apriori(dataset, minSupport, time_stamps=[], max_span=None, min_gap=None, max_gap=None, verbose=False):
    """
    The AprioriAll algorithm. Computes the frequent sequences in a seqeunce dataset for a given minSupport

    Args:
        dataset: A list of sequences, for which the frequent (sub-)sequences are computed
        minSupport: The minimum support that makes a sequence frequent
        verbose: If true, additional information on the mining process is printed (i.e., candidates on each level)
        time_stamps: Time stamps of the provided sequences. If empty, no time constraints will be applied.
        max_span: Maximum allowed time difference between latest and earliest events, expressed using `datetime.timedelta`.
        min_gap: Minimum time difference between two consecutive elements, expressed using `datetime.timedelta`.
        max_gap: Maximum time difference between two consecutive elements, expressed using `datetime.timedelta`.
    
    Returns:
        A list of tuples (s, c), where s is a frequent sequence, and c is the count for that sequence
    """
    # global numberOfCountingOperations
    # numberOfCountingOperations = 0
    Overall = []
    itemsInDataset = sorted(set([item for sublist1 in dataset for sublist2 in sublist1 for item in sublist2]))
    singleItemSequences = [[[item]] for item in itemsInDataset]
    singleItemCounts = [(i, countSupport(dataset, i)) for i in singleItemSequences if
                        countSupport(dataset, i) >= minSupport]
    Overall.append(singleItemCounts)
    if verbose:
        print("Result, lvl 1: " + str(Overall[0]))

    k = 1
    while (True):
        if not Overall[k - 1]:
            break
        # 1. Candidate generation
        candidatesLastLevel = [x[0] for x in Overall[k - 1]]
        candidatesGenerated = generateCandidates(candidatesLastLevel)
        # 2. Candidate pruning (using a "containsall" subsequences)
        generator = generateContiguousSubsequences if time_stamps else generateDirectSubsequences
        candidatesPruned = [cand for cand in candidatesGenerated if
                            all(x in candidatesLastLevel for x in generator(cand))]
        # 3. Support Counting
        candidatesCounts = [(i, countSupport(dataset, i, time_stamps, max_span, min_gap, max_gap)) for i in candidatesPruned]
        # 4. Candidate Elimination
        resultLvl = [(i, count) for (i, count) in candidatesCounts if (count >= minSupport)]
        if verbose:
            print("Candidates generated, lvl " + str(k + 1) + ": " + str(candidatesGenerated))
            print("Candidates pruned, lvl " + str(k + 1) + ": " + str(candidatesPruned))
            print("Result, lvl " + str(k + 1) + ": " + str(resultLvl))
        Overall.append(resultLvl)
        k = k + 1
    # "flatten" Overall
    Overall = Overall[:-1]
    Overall = [item for sublist in Overall for item in sublist]
    return Overall
