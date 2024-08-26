class RandomSampler {
    constructor(sampler) {
        this.sampler = sampler
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
    constructor(sampler, stepSize) {
        this.sampler = sampler
        this.stepSize = stepSize || 1
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
    constructor(string) {
        this.string = string
    }

    async take() {
        return this.string
    }
}

class DirectorySampler {
    constructor(directories) {
        this.directories = directories
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
    constructor(url) {
        this.url = 'https://www.gutenberg.org/files/100/old/shaks12.txt'
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
    constructor(sampler, stride) {
        this.sampler = sampler
        this.stride = stride || 0
        this.tokens = []
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
    constructor(samplers) {
        this.samplers = samplers
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
    constructor(samplers, rates) {
        this.samplers = samplers
        this.rates = rates
        this.currentIndex = 0
    }

    async take(config) {
        const i = this.currentIndex
        if (i + 1 >= this.samplers.length) this.currentIndex = 0
        else this.currentIndex++
        const roll = Math.random()
        if (roll < this.rates[i]) {
            return await this.samplers[i].take(config)
        } else {
            return await this.take(config)
        }
    }
}

class CosmopediaSampler {
    constructor(config) {
        this.config = config
        this.producer = null
    }

    async init() {
        const CosmopediaDataset = (await import('./datasets/cosmopedia.js'))
            .default
        this.producer = new CosmopediaDataset(this.config)
        await this.producer.init()
        this.initialized = true
    }

    async take(config) {
        if (!this.initialized) {
            await this.init()
        }
        return await this.producer.getSample({ size: config.maxSeqLen })
    }
}

class WikipediaSampler {
    constructor(config) {
        this.config = config
        this.producer = null
    }

    async init() {
        const WikipediaDataset = (await import('./datasets/wikipedia.js'))
            .default
        this.producer = new WikipediaDataset(this.config)
        await this.producer.init()
        this.initialized = true
    }

    async take(config) {
        if (!this.initialized) {
            await this.init()
        }
        return await this.producer.getSample({ size: config.maxSeqLen })
    }
}

class PhiSampler {
    constructor(config) {
        this.config = config
        this.producer = null
    }

    async init() {
        const PhiDataset = (await import('./datasets/phi.js')).default
        this.producer = new PhiDataset(this.config)
        await this.producer.init()
        this.initialized = true
    }

    async take(config) {
        if (!this.initialized) {
            await this.init()
        }
        if (config.isValidating) {
            return await this.producer.getSample({
                mode: 'validation',
                size: config.maxSeqLen
            })
        } else {
            return await this.producer.getSample({ size: config.maxSeqLen })
        }
    }
}

const samplers = {
    RandomSampler: (sampler) => new RandomSampler(sampler),
    SequentialSampler: (sampler, stepSize) =>
        new SequentialSampler(sampler, stepSize),
    StringSampler: (string) => new RandomSampler(new StringSampler(string)),
    DirectorySampler: (directories, delimiter) =>
        new RandomSampler(new DirectorySampler(directories, delimiter)),
    HTTPSampler: (url) => new RandomSampler(new HTTPSampler(url)),
    StridedSampler: (sampler, stride) => new StridedSampler(sampler, stride),
    CosmopediaSampler: (config) =>
        new StridedSampler(new CosmopediaSampler(config), 32),
    WikipediaSampler: (config) =>
        new StridedSampler(new WikipediaSampler(config), 32),
    PhiSampler: (config) => new StridedSampler(new PhiSampler(config), 32),
    MultiSampler: (samplers) => new MultiSampler(samplers),
    WeightedSampler: (samplers, rates) => new WeightedSampler(samplers, rates)
}
export default samplers
