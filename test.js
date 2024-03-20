import fs from 'fs'
import { shaks13 } from './src/data.js'

class BasicSubwordTokenizer {
    constructor(initialVocabulary, maxVocabSize = 32000) {
        this.maxVocabSize = maxVocabSize
        this.vocab = new Map(
            initialVocabulary.map((token, index) => [token, index])
        )
        this.tokenFrequencies = new Map()
        this.spaceToken = 'τ'
    }

    train(corpus, iterations = 10000, minLength = 2, maxLength = 7) {
        // Initial pass to handle spaces properly and ensure single characters are included
        for (const char of corpus) {
            if (char.trim() === '') {
                // Increment the space token's frequency for actual spaces
                this.tokenFrequencies.set(
                    this.spaceToken,
                    (this.tokenFrequencies.get(this.spaceToken) || 0) + 1
                )
            } else if (char) {
                // Exclude null or undefined characters
                this.tokenFrequencies.set(
                    char,
                    (this.tokenFrequencies.get(char) || 0) + 1
                )
            }
        }

        // Sample substrings from the entire corpus, avoiding leading or trailing spaces
        for (let i = 0; i < iterations; i++) {
            let start = Math.floor(Math.random() * (corpus.length - maxLength))
            let end =
                start +
                minLength +
                Math.floor(Math.random() * (maxLength - minLength))
            let token = corpus.substring(start, Math.min(end, corpus.length))

            // If the token contains spaces, we split it into words and sample individual words
            if (/\s/.test(token)) {
                const words = token.split(/\s+/)
                for (const word of words) {
                    // Exclude leading or trailing spaces
                    if (word.trim() !== '') {
                        // Increment the frequency of the word
                        this.tokenFrequencies.set(
                            word,
                            (this.tokenFrequencies.get(word) || 0) + 1
                        )
                    }
                }
            } else {
                // Remove any continuous spaces from the sampled token
                token = token.replace(/\s+/g, '')

                if (token) {
                    this.tokenFrequencies.set(
                        token,
                        (this.tokenFrequencies.get(token) || 0) + 1
                    )
                }
            }
        }

        this.finalizeVocabulary()
    }

    finalizeVocabulary() {
        const sortedTokens = Array.from(this.tokenFrequencies.entries())
            .filter(([token]) => token !== null && token !== undefined) // Exclude null or undefined tokens
            .sort((a, b) => b[1] - a[1])
            .map((entry) => entry[0])
            .slice(0, this.maxVocabSize - this.vocab.size)

        sortedTokens.forEach((token, index) => {
            if (!this.vocab.has(token)) {
                this.vocab.set(token, this.vocab.size)
            }
        })
    }

    writeVocabularyToFile(filePath) {
        const vocabArray = Array.from(this.vocab.keys()) // Only need keys for the vocab file
        const vocabJson = JSON.stringify(vocabArray, null, 2)
        fs.writeFileSync(filePath, vocabJson, 'utf8')
        console.log(`Vocabulary written to ${filePath}`)
    }

    encode(text) {
        const tokens = []
        let currentToken = ''
        for (let i = 0; i < text.length; i++) {
            const char = text[i]
            const potentialToken = currentToken + char
            if (this.vocab.has(potentialToken)) {
                currentToken = potentialToken
            } else {
                if (currentToken) tokens.push(this.vocab.get(currentToken))
                currentToken = this.vocab.has(char) ? char : this.spaceToken
            }
        }
        if (currentToken) tokens.push(this.vocab.get(currentToken))
        return tokens.filter((token) => token !== undefined)
    }
}

// Example usage
const initialVocab =
    `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `.split(
        ''
    )
const tokenizer = new BasicSubwordTokenizer(initialVocab, 1000)

// Assuming `largeText` is the string you want to tokenize
tokenizer.train(shaks13, 10000000, 1, 7)
console.log(tokenizer.vocab)
tokenizer.writeVocabularyToFile('./vocabulary.json')
const encodedText = tokenizer.encode('Hello, world!')
console.log(encodedText)
