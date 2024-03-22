import { Llama2Tokenizer } from '@lenml/llama2-tokenizer'
import { load_vocab } from '@lenml/llama2-tokenizer-vocab-llama2'
import { shaks13 } from './data.js'

export class BasicSubwordTokenizer {
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
        console.log('training a tokenizer')
        this.train(corpus, trainIterations, 1, 7)
    }

    train(corpus, maxIterations = 10_000_000, minLength = 2, maxLength = 7) {
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
                this.finalizeVocabulary()
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

    async writeVocabularyToFile(path = './data/models/ode') {
        if (typeof window === 'undefined') {
            const vocabArray = Array.from(this.vocab.keys())
            const vocabJson = JSON.stringify(vocabArray, null, 2)
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

// export class BasicSubwordTokenizer {
//     constructor(maxVocabSize = 32000, corpus = shaks13) {
//         this.maxVocabSize = maxVocabSize
//         const initialVocab =
//             `¶0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.?!&'"\`;:(){}[]<>#*^%$@~+-=_|/\\\n `.split(
//                 ''
//             )
//         this.vocab = new Map(initialVocab.map((token, index) => [token, index]))

//         this.tokenFrequencies = new Map()
//         this.spaceToken = 'τ'
//         console.log('training a tokenizer')
//         this.train(corpus, 10_000_000, 100_000_000, 1, 7)
//         console.log(Array.from(this.vocab).length)
//         console.log(this.maxVocabSize)
//     }

//     train(
//         corpus,
//         minIterations = 1_000_000,
//         maxIterations = 10_000_000,
//         minLength = 2,
//         maxLength = 7
//     ) {
//         let vocabSize = this.vocab.size
//         let iterationCounter = 0 // Adjusted to count the total number of samples

//         while (
//             vocabSize < this.maxVocabSize &&
//             iterationCounter < maxIterations
//         ) {
//             let start = Math.floor(Math.random() * (corpus.length - maxLength))
//             let len =
//                 minLength +
//                 Math.floor(Math.random() * (maxLength - minLength + 1))
//             let token = corpus.substring(start, start + len)

//             // Check if token contains space, and only process the part before the space
//             if (token.includes(' ')) {
//                 token = token.substring(0, token.indexOf(' '))
//             }

//             if (token && !this.vocab.has(token)) {
//                 this.tokenFrequencies.set(
//                     token,
//                     (this.tokenFrequencies.get(token) || 0) + 1
//                 )
//             }

//             iterationCounter++
//             if (iterationCounter % minIterations === 0) {
//                 // Log every 100,000 iterations for progress tracking
//                 console.log(
//                     `Iteration ${iterationCounter}: Current vocab size is ${this.vocab.size}`
//                 )
//             }

//             // Only finalize and check vocab size after a significant number of samples
//             if (
//                 iterationCounter % minIterations === 0 ||
//                 iterationCounter === maxIterations
//             ) {
//                 this.finalizeVocabulary()
//                 if (this.vocab.size > vocabSize) {
//                     vocabSize = this.vocab.size
//                 } else {
//                     // If no new tokens were added after a round of sampling, consider stopping early
//                     console.log(
//                         `No new tokens found. Stopping early after ${iterationCounter} iterations.`
//                     )
//                     break
//                 }
//             }
//         }

//         console.log(
//             `Final vocab size: ${this.vocab.size} after ${iterationCounter} iterations.`
//         )
//     }

//     finalizeVocabulary() {
//         const neededTokens = this.maxVocabSize - this.vocab.size
//         const sortedTokens = Array.from(this.tokenFrequencies.entries())
//             .sort((a, b) => b[1] - a[1])
//             .slice(0, neededTokens)

//         sortedTokens.forEach(([token]) => {
//             if (!this.vocab.has(token)) {
//                 this.vocab.set(token, this.vocab.size)
//             }
//         })
//     }

//     async writeVocabularyToFile(path = './data/models/ode') {
//         if (typeof window === 'undefined') {
//             const vocabArray = Array.from(this.vocab.keys()) // Only need keys for the vocab file
//             const vocabJson = JSON.stringify(vocabArray, null, 2)
//             const fs = await import('fs')
//             fs.mkdirSync(path, { recursive: true })
//             fs.writeFileSync(`${path}/tokenizer.json`, vocabJson, 'utf8')
//             console.log(`Vocabulary written to ${path}`)
//         }
//     }

//     getLength() {
//         return this.vocab.size
//     }

//     decode(tokens) {
//         let decodedText = ''
//         let currentToken = ''

//         for (const token of tokens) {
//             let tokenStr = ''
//             if (typeof token === 'number') {
//                 tokenStr = Array.from(this.vocab.keys())[token]
//             } else {
//                 tokenStr = token
//             }

//             if (tokenStr === this.spaceToken) {
//                 decodedText += ' '
//             } else {
//                 currentToken += tokenStr
//                 if (this.vocab.has(currentToken)) {
//                     decodedText += currentToken
//                     currentToken = ''
//                 }
//             }
//         }

//         return decodedText
//     }

//     encode(text) {
//         const tokens = []
//         let currentToken = ''
//         for (let i = 0; i < text.length; i++) {
//             const char = text[i]
//             const potentialToken = currentToken + char
//             if (this.vocab.has(potentialToken)) {
//                 currentToken = potentialToken
//             } else {
//                 if (currentToken) tokens.push(this.vocab.get(currentToken))
//                 currentToken = this.vocab.has(char) ? char : this.spaceToken
//             }
//         }
//         if (currentToken) tokens.push(this.vocab.get(currentToken))
//         return tokens.filter((token) => token !== undefined)
//     }
// }

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
