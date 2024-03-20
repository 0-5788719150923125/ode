import { Llama2Tokenizer } from '@lenml/llama2-tokenizer'
import { load_vocab } from '@lenml/llama2-tokenizer-vocab-llama2'
import { shaks13 } from './data.js'

export class BasicSubwordTokenizer {
    constructor(corpus = shaks13, maxVocabSize = 8888) {
        this.maxVocabSize = maxVocabSize
        const initialVocab =
            `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `.split(
                ''
            )
        this.vocab = new Map(initialVocab.map((token, index) => [token, index]))

        this.tokenFrequencies = new Map()
        this.spaceToken = 'τ'
        console.log('training a tokenizer')
        this.train(corpus, 10000000, 1, 7)
        console.log(Array.from(this.vocab).length)
        console.log(this.maxVocabSize)
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

    async writeVocabularyToFile(filePath = './data/tokenizer.json') {
        if (typeof window === 'undefined') {
            const fs = await import('fs')
            const vocabArray = Array.from(this.vocab.keys()) // Only need keys for the vocab file
            const vocabJson = JSON.stringify(vocabArray, null, 2)
            fs.writeFileSync(filePath, vocabJson, 'utf8')
            console.log(`Vocabulary written to ${filePath}`)
        }
    }

    getLength() {
        return this.maxVocabSize
    }

    decode(tokens) {
        let decodedText = ''
        let currentToken = ''

        for (const token of tokens) {
            let tokenStr = ''
            if (typeof token === 'number') {
                tokenStr = Array.from(this.vocab.keys())[token]
            } else {
                tokenStr = token
            }

            if (tokenStr === this.spaceToken) {
                decodedText += ' '
            } else {
                currentToken += tokenStr
                if (this.vocab.has(currentToken)) {
                    decodedText += currentToken
                    currentToken = ''
                }
            }
        }

        return decodedText
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

export default class Tokenizer {
    constructor() {
        // this.padToken = '�'
        // this.vocab = Array.from(
        //     new Set(
        //         `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
        //     )
        // )
        // this.vocab.unshift(this.padToken)
        this.model = new Llama2Tokenizer()
        this.model.install_vocab(load_vocab())
    }

    getLength() {
        return this.model.vocab_size
    }

    encode(string) {
        return this.model.encode(string)
    }

    decode(sequence) {
        return this.model.decode(sequence)
    }
}
