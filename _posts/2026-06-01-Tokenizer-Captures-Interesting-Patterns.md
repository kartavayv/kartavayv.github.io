---
layout: post
title: Tokenizer Captures Interesting Patterns
date: 2026-06-01
categories: Questions
tags: ["Machine Learning", "Natural Language Processsing"]
---
#TODO: calling them "byte combinations" doesn't seem like a good idea!!!

I was working on making a very simple tokenizer recently. Tokenizers just convert a lot of words into a sequence of numbers which we can call tokens. The reason they are so important is because most of the AI today is about tokens, remove them from the line you have no ChatGPT or Claude or even... I don't know.  

The idea behind tokenization is very simple. You have a sequence of words like: ``The music is loud. I am even louder.`` This small chunk of text may get converted to something like: ``[84, 104, 256, 109, 117, 283, 99, 32, 272, 108, 265, 100, 379, 260, 309, 451, 108, 265, 100, 262, 46]``. These numbers are tokens, an LLM sees a text in the form of a token, which are represented as vectors, but we will concern ourselves with only the token part in this post.

The first thing about generating tokens is knowing how to generate them. Bytes are very cool things, aren't they?. There are 8 bits in a byte, we can make 256 different permutations of those bits, so it seems like a good idea to represent characters using these various permutations of bits. 256 different permutations of bits can be used to represent 256 different characters. Byte 1 might represent one thing and byte 65 other. We are not the first one to think of this, sadly! There is something called UTF-8 which does exactly this, maps permutations of bits to various characters. Python has functions ``encode`` and ``decode`` which use UTF-8 (can also use other unicode formats) to encode or decode a character into its byte combination.  

```python
"😊".encode()   #results in: b'\xf0\x9f\x98\x8a'

b'\xf0\x9f\x98\x8a'.decode()    #😊 (we get the emoji back)

"नमस्ते".encode()     #languages other than english, take a lot of bytes: b'\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87'

```  
So, with basic ``encode`` and ``decode`` we have a decent base to build our tokenizer. For a large chunk of text we can get a large sequence of tokens which is not a good idea to train our model on. We need a method of compression. A simple approach is by looking at the sequence of "fresh" tokens we get for a string and look up for repeatedly occuring pairs of tokens in it. Then start replacing these pairs with an integer >255 since integers in range [0,255] represent those 256 byte combinations we talked about. We can add as many new tokens as we like which will just be the combinations of the first 256 tokens, just occuring together in the training data by chance.  

```python
#Looks for the pairs (2) of tokens in the token sequence and total number of times it occurs in the sequence

    def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) +1

    return counts

```  
To compress the raw sequence of tokens:  

```python
    def merge(self,ids, pair, idx):
        max_len = len(ids)
        new_ids = []
        i = 0

        while i < max_len :
            if i< max_len -1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        
        return new_ids
```  
To generate the vocabulary of our custom tokenizer we need to loop on the token sequence using ``func: merge``. One thing to keep in mind is after the first loop the compressed sequence will be different from the raw sequence we get from default ``.encode()`` of Python. We need to run the subsequent loops on this sequence which repeatedly gets compressed. Simple training code:  

```python

    def generate_vocab(self):
        new_vocab = 500
        num_itr = new_vocab - 256
        ids = list(self.tokens)

        for i in range(num_itr):
            stat = self.stats(ids)
            pair = max(stat, key=stat.get)
            idx = i + 256
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            # print(f"merged token {pair} -> {idx}") #Try printing it out!
```  

Here, new_vocab - 256, is the actual number of new tokens we are adding to our tokenizer vocabulary, we can call them merge tokens. Take a look at the merged bytes we got from the training text:  

<figure align="centre">
  <img src = "/assets/images/2026-06-10-Tokenizer-Captures-Interesting-Patterns/june_blog_2026.png">
</figure>  

Training text I used (from one of my blog posts, sort of philosophical):  

``` 
text = """eath all the great stories, myths, and even hit movies of our time. Storylines can be different, but they follow the same cycle. Beyond myths, it’s the way we learn in our lives; the more cycles a person goes through, the more he come to know himself. When life stinks, that’s the call for another hero journey.

# Distractions easily numb the pain in our lives and, as a result, make us deaf to the calling for our next hero journey. For Joseph Campbell, a good life is one hero journey after the other. It’s the radical acceptance of the adventure that life throws at you.

# “ What I think is that a good life is one hero journey after another. Over and over again, you are called to the realm of adventure; you are called to new horizons. Each time, there is the same problem: do I dare? And then if you do dare, the dangers are there, and the help also, and the fulfillment or the fiasco. There’s always the possibility of a fiasco “ ~ Joseph Campbell

# Fear of Death
# You can find the same pattern in most of the stories of the culture, just the characters and story plots change in the cultural context. Hero was living a cozy life, gets into trouble, deals with it, faces an ordeal, and comes back more mature and stronger.

# Transcending death is a part of the hero journey. It’s a triumph over the biological limits of man; it’s precisely what we secretly crave for in our love for heroes, and the opposite is what we are dangerously afraid of.

# We are, in short, narcissists by design. For us, the whole world revolves around us, and still, we are not satisfied. I found this in Denial of Death, and this explains our narcissistic nature well: Luck is when the guy next to you gets hit with the arrow.

# Apart from this “narcissistic” nature of ours, we are also blessed with self-image. We don’t merely live in the virtual physical world, but also, to be more precise, all the time, a large portion of our attention goes into dealing with the abstract world inside our heads.

# This passage from Denial of Death makes it even clearer:

# “In man, a working level of narcissism is inseparable from self-esteem, from a basic sense of self-worth. We have learned, mostly from Alfred Adler, that what man needs most is to feel secure in his self-esteem.”

# Our self-esteem feeds on the abstractions like sound, images, and words floating in our memory that associate back to us. A depressed man most likely has sounds, images, and words floating in his head, all the time, that scream to him that he is worth of a housefly. The abstractions self-esteem stands upon are what make a depressed or a normal person, and this influences our narcissism. In the center of the perception of both a depressed and a normal person is himself, that is, both are narcissists.

# When this narcissistic agent meets death, he cannot believe that he, who is supreme, the protagonist, the main character of his w"""

```  
Real world models will obviously be a lot larger. I remember reading in context to a certain GPT model, that it has a vocabulary of 100,000 words that's a lot larger than we are using in this pedagogical example.  

After learning the vocabulary, we need to make our custom ``encode`` and ``decode`` functions, because the baseline functions that Python gives us can't use the vocabulary that we have learnt.   

```python

    @staticmethod
    def vocab_bytes(merges):
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0,p1), v in merges.items():
            vocab[v] = vocab[p0] + vocab[p1]
        
        return vocab

    def decode(self, ids):
        vocab = self.vocab_bytes(self.merges)
        tokens = b"".join([vocab[idx] for idx in ids])
        text = tokens.decode("utf-8", errors="replace")

        return text
    
    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stat = self.stats(tokens)
            pair = min(stat, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])
        
        return tokens

```

Our encode function, takes a chunk of text in and then converts them into bytes and gets a list of their integer representations. We then try to compress this sequence using our learnt vocabulary. I personally find decode function very facinating, it takes in a sequence of tokens, converts them into a byte sequence and then we just apply default ``decode`` Python provides us. I really like that byte sequence thing, such a cutie!  

Also I encoded a random text chunk, I really like how certain tokens have come closer to representing parts of words and sometimes even a full word.  

<figure align="centre">
  <img src = "/assets/images/2026-06-10-Tokenizer-Captures-Interesting-Patterns/june2026_blog_tokenizer_2.png">
</figure>  

<figure align="centre">
  <img src = "/assets/images/2026-06-10-Tokenizer-Captures-Interesting-Patterns/june2026_blog_tokenizer_3.png">
</figure>  

The more runs and training text I give to it, the better it can do this, for example, now with increased runs:  
<figure align="centre">
  <img src = "/assets/images/2026-06-10-Tokenizer-Captures-Interesting-Patterns/june2026_blog_tokenizer_4.png">
</figure>  
<figure align="centre">
  <img src = "/assets/images/2026-06-10-Tokenizer-Captures-Interesting-Patterns/june2026_blog_tokenizer.png">
</figure>  

Today we have some really advanced (in functionality) tokenizers, which differ and use a different way of making tokens (some GPT models use regex in their tokenizer). I plan to implement them in future and possibly update this blogpost with the updated knowledge and intuition as a result.
