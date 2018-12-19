---
title: "Adding and Searching a Trie"
date: "2018-12-19"
path: "/trie/"
category: "Notes"
section: "Algorithms And Data Structures"
---


Design a data structure that supports the following two operations:

```
void addWord(word)
bool search(word)
```

`search(word)` can search a literal word or a regular expression string containing only letters `a-z` or `.`. A `.` means it can represent any one letter.

Example:
```
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
```
Note:
You may assume that all words are consist of lowercase letters a-z.

## Approach

We're going to define a `trieNode` data structure that holds a dictionary of next letters. The next letters will be trie nodes themselves. Each trie node will have a property `end`, which will denote whether that node marks the end of a word.


### Adding a Word
The algorithm starts with the root node and the first letter of the word. If the letter is not a child of the node, add a `trieNode` of the letter to the nodes children, and update the current node to be this child. When there are no more letters left, mark the current nodes `end` property to be `True`.

### Searching for a Word
The algorithm starts with the root node and the word. If there is no word, then return the `end` property of the node, signaling whether this node marks the end of a complete word or not. 

Otherwise, get the first letter of the word. There are two cases to solve for:

**1) Letter is a character**

- check if letter is in the nodes children, and check if subsequent search of the remaining word is in that child node.

**2) Letter is a period**
- check if subsequent search of the remaining word returns true for any of the children nodes.


```python
from collections import defaultdict
import string


class TrieNode:
    
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.end = False
        
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        self.alphabet = list(string.ascii_lowercase)
        
    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        curr = self.root
        
        for letter in word:
            curr = curr.children[letter]
        curr.end = True
        

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain 
        the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        return self._search(word, self.root)
    
    def _search(self, word, node):
        
        if not word:
            return node.end
        else:
            char, word = word[0], word[1:]
            
            if char != '.':
                return char in node.children and self._search(word, node.children[char])
            else:
                return any([self._search(word, child) for child in node.children.values()])

```
