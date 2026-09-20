import regex as re
import os
from typing import BinaryIO

PRE_TOKENIZER_PAT  = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>"]
def train_bpe(input_path, vocab_size, special_tokens):
    """
    Train a BPE tokenizer on the input file.
    """
    SPECIAL_TOKENS_PAT = "|".join(re.escape(token) for token in special_tokens)
    word_freq = {}
    pair_count = {}
    vocab = dict()
    merges = []
    for id in range(256):
        vocab[id] = bytes([id])

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            for article in re.split(SPECIAL_TOKENS_PAT, chunk):
                for match in re.finditer(PRE_TOKENIZER_PAT, article):
                    word = match.group().encode("utf-8")
                    word_freq[tuple(word)] = word_freq.get(tuple(word), 0) + 1

    #print(word_freq)

    for k, v in word_freq.items():
        for i in range(len(k) - 1):
            pair = (k[i], k[i + 1])
            pair_count[pair] = pair_count.get(pair, 0) + v

    #print(pair_count)
    #print(max(pair_count.values()))
    merge_time = 0

    while merge_time < vocab_size - 256:
        #sort and pick up the most frequent pair
        max_pair = max(pair_count, key=pair_count.get)
        #print(max_pair)

        vocab[len(vocab)] = vocab[max_pair[0]] + vocab[max_pair[1]]
        #print(vocab[256])
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
        del(pair_count[max_pair])
        merge_time += 1

        # search merged pair in word_freq and update the frequency
        new_word_freq = {}
        for k, v in word_freq.items():
            new_k = ()
            i = 0
            merge_pos = []
            while i < len(k):
                if i<len(k)-1 and (k[i], k[i + 1]) == max_pair:
                    new_k += (len(vocab)-1,)
                    merge_pos.append(len(new_k)-1)
                    i += 2
                else:
                    new_k += (k[i],)
                    i += 1
            #update pair_count
            for idx in merge_pos:
                if idx > 0:
                    left_pair = (new_k[idx-1], new_k[idx])
                    pair_count[left_pair] = pair_count.get(left_pair, 0) + v
                if idx < len(new_k)-1:
                    right_pair = (new_k[idx], new_k[idx+1])
                    pair_count[right_pair] = pair_count.get(right_pair, 0) + v
            new_word_freq[new_k] = v

        word_freq = new_word_freq
    return vocab, merges

                
#train_bpe("input.txt", 256, ["<|endoftext|>"])

# test = '''
# How UTF-8 Works

# UTF-8 encodes each Unicode character into a sequence of one to four bytes. The first 128 characters in the Unicode library, which correspond to ASCII characters, are encoded using a single byte. Characters that appear later in the Unicode library are encoded as two-byte, three-byte, and eventually four-byte binary units.

# Here is a table showing how different characters are encoded in UTF-8:

# | Character | Code Point | UTF-8 Encoding | |-----------|------------|-------------------------| | A | U+0041 | 01000001 | | Ø | U+00D8 | 11000011 10011000 | | 𠜎 | U+2070E | 11110000 10100000 10011100 10001110 | | 😁 | U+1F601 | 11110000 10011111 10011000 10000001 |
# '''



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

vocab, merges = train_bpe(r"C:\Users\wishf\OneDrive\Desktop\projects\llm\data\TinyStoriesV2-GPT4-valid.txt", 500, SPECIAL_TOKENS)

print("Vocabulary:")
for id, token in vocab.items():
    print(f"{id}: {token}")

print("\nMerges:")
for left, right in merges:
    print(f"{left} + {right} -> {left + right}")