import { Llama2Tokenizer } from '@lenml/llama2-tokenizer'
import { load_vocab } from '@lenml/llama2-tokenizer-vocab-llama2'
import { shaks13 } from './data.js'

export class BasicSubwordTokenizer {
    constructor(corpus = shaks13, maxVocabSize = 6666) {
        this.maxVocabSize = maxVocabSize
        const initialVocab =
            `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `.split(
                ''
            )
        this.vocab = new Map(initialVocab.map((token, index) => [token, index]))

        this.tokenFrequencies = new Map()
        this.spaceToken = 'τ'
        console.log('training a tokenizer')
        this.train(corpus, 10_000_000, 1, 7)
        console.log(Array.from(this.vocab).length)
        console.log(this.maxVocabSize)
    }

    train(corpus, maxIterations = 10_000_000, minLength = 2, maxLength = 7) {
        let vocabSize = this.vocab.size
        let newTokensFound = true
        let iterationCounter = 0 // Added counter to manage logging

        while (vocabSize < this.maxVocabSize && newTokensFound) {
            newTokensFound = false
            for (let i = 0; i < corpus.length - maxLength; i++) {
                for (let len = minLength; len <= maxLength; len++) {
                    let token = corpus.substring(i, i + len).replace(/\s+/g, '')
                    if (token && !this.vocab.has(token)) {
                        this.tokenFrequencies.set(
                            token,
                            (this.tokenFrequencies.get(token) || 0) + 1
                        )
                        newTokensFound = true
                    }
                }
            }

            this.finalizeVocabulary()
            console.log('boop')
            if (this.vocab.size > vocabSize) {
                vocabSize = this.vocab.size
                // Logging after each vocabulary update
                console.log(`Current vocab length is: ${vocabSize}`)
                console.log(
                    `Trained tokenizer in ${iterationCounter} iterations`
                )
            } else {
                // Break the loop if no new tokens were added in this iteration
                break
            }

            iterationCounter++
            // Additional logging condition, adjust as needed for frequency
            if (iterationCounter % 10 === 0) {
                console.log(
                    `Iteration ${iterationCounter}: vocab size is ${vocabSize}`
                )
            }
        }
    }

    finalizeVocabulary() {
        const neededTokens = this.maxVocabSize - this.vocab.size
        const sortedTokens = Array.from(this.tokenFrequencies.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, neededTokens)

        sortedTokens.forEach(([token]) => {
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
