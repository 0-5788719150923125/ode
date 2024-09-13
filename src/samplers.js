import RefinedWebDataset from './datasets/refinedweb.js'
import CosmopediaDataset from './datasets/cosmopedia.js'
import WikipediaDataset from './datasets/wikipedia.js'
import PhiDataset from './datasets/phi.js'
import { LinearCongruentialGenerator } from './utils.js'

class RandomSampler {
    constructor(config) {
        this.sampler = config.sampler
        this.tokens = []
    }

    async take(config) {
        const string = await this.sampler.take(config)
        const expandedLen = config.maxSeqLen * 10
        const startIndex = Math.floor(
            Math.random() * (string.length - expandedLen)
        )
        const sample = string.substring(startIndex, startIndex + expandedLen)
        const tokens = config.tokenizer.encode(sample)
        return tokens.slice(0, config.maxSeqLen)
    }
}

class SequentialSampler {
    constructor(config) {
        this.sampler = config.sampler
        this.stepSize = config?.stepSize || 1
    }

    async take(config) {
        const string = await this.sampler.take(config)
        if (!this.index) {
            this.index = Math.floor(
                Math.random() * (str.length - sampleLen + 1)
            )
        }
        if (this.index + config.maxSeqLen > string.length) {
            this.index = 0
        }

        const expandedLen = config.maxSeqLen * 10
        const useIndex = this.index
        this.index = this.index + this.stepSize
        return string.substring(useIndex, useIndex + expandedLen)
    }
}

class StringSampler {
    constructor(config) {
        this.string = config.string
    }

    async take() {
        return this.string
    }
}

class DirectorySampler {
    constructor(config) {
        this.directories = config.directories
    }

    async read({ delimiter = '\n\n' } = {}) {
        const fs = (await import('fs')).default
        const path = (await import('path')).default

        let allText = ''

        const isValidUtf8 = (buffer) => {
            return buffer.every((byte) => byte <= 127)
        }

        const readDirSync = (dir) => {
            const entries = fs.readdirSync(dir, { withFileTypes: true })
            for (const entry of entries) {
                const entryPath = path.join(dir, entry.name)
                if (entry.isDirectory()) {
                    readDirSync(entryPath)
                } else {
                    const fileBuffer = fs.readFileSync(entryPath)
                    if (isValidUtf8(fileBuffer)) {
                        const fileContent = fileBuffer.toString('utf8')
                        if (fileContent.trim() !== '') {
                            allText += `${fileContent}${delimiter}`
                        }
                    }
                }
            }
        }

        const directories = this.directories.split(',')

        for (const directory of directories) {
            readDirSync(directory)
        }

        this.string = allText
    }

    async take(config) {
        if (!this.string) await this.read(config)
        return this.string
    }
}

class HTTPSampler {
    constructor(config) {
        this.url =
            config?.url || 'https://www.gutenberg.org/files/100/old/shaks12.txt'
    }

    async read() {
        const response = await fetch(this.url)
        this.string = await response.text()
    }

    async take() {
        if (!this.string) await this.read()
        return this.string
    }
}

class StridedSampler {
    constructor(config) {
        this.sampler = config.sampler
        this.stride = config?.stride || 0
        this.tokens = []
    }

    async init() {
        await this.sampler.init()
        this.initialized = true
    }

    resetGenerator(mode = 'train') {
        this.sampler.resetGenerator(mode)
    }

    async take({ tokenizer, maxSeqLen, isValidating = false } = {}) {
        while (true) {
            if (this.tokens.length >= maxSeqLen) {
                const returnTokens = this.tokens.slice(0, maxSeqLen)
                this.tokens = this.tokens.slice(maxSeqLen - this.stride)
                return returnTokens
            }
            const sample = await this.sampler.take({
                tokenizer,
                maxSeqLen,
                isValidating
            })
            this.tokens.push(...tokenizer.encode(sample))
        }
    }
}

class MultiSampler {
    constructor(config) {
        this.samplers = config.samplers
        this.currentIndex = 0
    }

    async take(config) {
        const i = this.currentIndex
        if (i + 1 >= this.samplers.length) this.currentIndex = 0
        else this.currentIndex++
        return await this.samplers[i].take(config)
    }
}

class WeightedSampler {
    constructor(config) {
        this.samplers = config.samplers
        this.rates = config.rates
        this.currentIndex = 0
        if (config?.seed) {
            const lcg = new LinearCongruentialGenerator(config.seed)
            this.randomFloat = (...args) => lcg.randomFloat(...args)
        } else {
            this.randomFloat = Math.random
        }
    }

    async take(config) {
        if (config?.isValidating) {
            // This is a super hack, but I'm tired
            const i = 1 // The Phi dataset
            return await this.samplers[i].take(config)
        }

        const i = this.currentIndex
        if (i + 1 >= this.samplers.length) this.currentIndex = 0
        else this.currentIndex++
        const roll = this.randomFloat(0, 1)
        if (roll < this.rates[i]) {
            return await this.samplers[i].take(config)
        } else {
            return await this.take(config)
        }
    }

    resetGenerator(mode = 'train') {
        // This is part of the same super hack
        this.samplers[1].resetGenerator(mode)
    }
}

class HuggingFaceSampler {
    constructor(config) {
        this.config = config
        const Dataset = config.dataset
        this.sampler = new StridedSampler({
            ...config,
            sampler: new Dataset(this.config),
            stride: 64
        })
    }

    async init() {
        await this.sampler.init()
        this.initialized = true
    }

    async take(config) {
        if (!this.initialized) await this.init()
        return await this.sampler.take({
            ...config,
            mode: config.isValidating ? 'validation' : 'train'
        })
    }
}

class RefinedWebSampler extends HuggingFaceSampler {
    constructor(config) {
        super({ ...config, dataset: RefinedWebDataset })
    }
}

class CosmopediaSampler extends HuggingFaceSampler {
    constructor(config) {
        super({ ...config, dataset: CosmopediaDataset })
    }
}

class WikipediaSampler extends HuggingFaceSampler {
    constructor(config) {
        super({ ...config, dataset: WikipediaDataset })
    }
}

class PhiSampler extends HuggingFaceSampler {
    constructor(config) {
        super({ ...config, dataset: PhiDataset })
    }
}

export default {
    RandomSampler: (config) => new RandomSampler(config),
    SequentialSampler: (config) => new SequentialSampler(config),
    StringSampler: (config) =>
        new RandomSampler({ sampler: new StringSampler(config) }),
    DirectorySampler: (config) =>
        new RandomSampler({
            sampler: new DirectorySampler(config)
        }),
    HTTPSampler: (config) =>
        new RandomSampler({ sampler: new HTTPSampler(config) }),
    StridedSampler: (config) => new StridedSampler(config),
    MultiSampler: (config) => new MultiSampler(config),
    WeightedSampler: (config) => new WeightedSampler(config),
    CosmopediaSampler: (config) => new CosmopediaSampler(config),
    WikipediaSampler: (config) => new WikipediaSampler(config),
    RefinedWebSampler: (config) => new RefinedWebSampler(config),
    PhiSampler: (config) => new PhiSampler(config)
}
