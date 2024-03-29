import { shaks13 } from './data.js'

class CharacterTokenizer {
    constructor() {
        this.padToken = '�'
        this.vocab = Array.from(
            new Set(
                `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `
            )
        )
        this.vocab.unshift(this.padToken)
        this.model = new Llama2Tokenizer()
        this.model.install_vocab(load_vocab())
    }

    getLength() {
        return this.vocab.length
    }

    encode(string) {
        // not implemented
    }

    decode(array) {
        // not implemented
    }

    writeVocabularyToFile(path) {
        // skip
    }
}

import { env } from '@xenova/transformers'
env.allowLocalModels = false
import { AutoTokenizer } from '@xenova/transformers'
class XenovaTokenizer {
    constructor(config) {
        this.model = config.model || 'openai-community/gpt2'
        this.tokenizer
    }

    async init() {
        this.tokenizer = await AutoTokenizer.from_pretrained(this.model)
    }

    getLength() {
        return this.tokenizer.model.vocab.length
    }

    encode(string) {
        return this.tokenizer.encode(string)
    }

    decode(array) {
        return this.tokenizer.decode(array, { skip_special_tokens: true })
    }

    writeVocabularyToFile(path) {
        // skip
    }
}

class BasicSubwordTokenizer {
    constructor(
        maxVocabSize = 32000,
        trainIterations = 50_000_000,
        corpus = shaks13
    ) {
        this.maxVocabSize = maxVocabSize
        const initialVocab =
            `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `.split(
                ''
            )
        this.vocab = new Map(initialVocab.map((token, index) => [token, index]))
        this.tokenFrequencies = new Map()
        this.train(corpus, trainIterations, 1, 7)
    }

    train(corpus, maxIterations = 10_000_000, minLength = 2, maxLength = 7) {
        console.log('training a tokenizer')

        let vocabSize = this.vocab.size
        let iterationCounter = 0

        while (
            vocabSize < this.maxVocabSize &&
            iterationCounter < maxIterations
        ) {
            let start = Math.floor(Math.random() * (corpus.length - maxLength))
            let len =
                minLength +
                Math.floor(Math.random() * (maxLength - minLength + 1))
            let token = corpus.substring(start, start + len)

            if (token && !this.vocab.has(token)) {
                this.tokenFrequencies.set(
                    token,
                    (this.tokenFrequencies.get(token) || 0) + 1
                )
            }

            iterationCounter++
            if (iterationCounter % 10_000_000 === 0) {
                console.log(
                    `Iteration ${iterationCounter}: Current vocab size is ${this.tokenFrequencies.size}`
                )
            }

            if (iterationCounter === maxIterations) {
                this.mergeVocabulary()
                if (this.vocab.size > vocabSize) {
                    vocabSize = this.vocab.size
                } else {
                    console.log(
                        `No new tokens found. Stopping early after ${iterationCounter} iterations.`
                    )
                    break
                }
            }
        }

        console.log(
            `Final vocab size: ${this.vocab.size} after ${iterationCounter} iterations.`
        )
    }

    mergeVocabulary() {
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

    async writeVocabularyToFile(path = './data/models/ode') {
        if (typeof window === 'undefined') {
            const vocabJson = JSON.stringify(
                Array.from(this.vocab.keys()),
                null,
                2
            )
            const fs = await import('fs')
            fs.mkdirSync(path, { recursive: true })
            fs.writeFileSync(`${path}/tokenizer.json`, vocabJson, 'utf8')
            console.log(`Vocabulary written to ${path}`)
        }
    }

    getLength() {
        return this.vocab.size
    }

    decode(tokens) {
        return tokens
            .map((token) =>
                typeof token === 'number'
                    ? Array.from(this.vocab.keys())[token]
                    : token
            )
            .join('')
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
                currentToken = this.vocab.has(char) ? char : ''
            }
        }
        if (currentToken) tokens.push(this.vocab.get(currentToken))
        return tokens.filter((token) => token !== undefined)
    }
}

const tokenizers = {
    CharacterTokenizer: (config) => new CharacterTokenizer(config),
    BasicSubwordTokenizer: (maxVocabSize, trainIterations, corpus) =>
        new BasicSubwordTokenizer(maxVocabSize, trainIterations, corpus),
    XenovaTokenizer: (config) => new XenovaTokenizer(config)
}
export default tokenizers
