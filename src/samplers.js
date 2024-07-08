// function* stringSampler(sampleLen, overfit = 0, str = '') {
//     if (overfit > 0) str = splitLines(str, overfit)
//     while (true) {
//         // Generate a random start index within the string's bounds
//         const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
//         // Yield a ${sampleLen} substring
//         yield str.substring(startIndex, startIndex + sampleLen)
//     }
// }

function* sequentialStringSampler(sampleLen, str) {
    let index = Math.floor(Math.random() * (str.length - sampleLen + 1))
    while (true) {
        if (index + sampleLen > str.length) {
            index = 0
        }
        yield str.substring(index, index + sampleLen) // Yield a substring of length sampleLen
        index++
    }
}

// async function directorySampler(dirs = './', delimiter = '\n\n') {
//     const fs = (await import('fs')).default
//     const path = (await import('path')).default

//     let allText = ''

//     const isValidUtf8 = (buffer) => {
//         return buffer.every((byte) => byte <= 127)
//     }

//     const readDirSync = (dir) => {
//         const entries = fs.readdirSync(dir, { withFileTypes: true })
//         for (const entry of entries) {
//             const entryPath = path.join(dir, entry.name)
//             if (entry.isDirectory()) {
//                 readDirSync(entryPath)
//             } else {
//                 const fileBuffer = fs.readFileSync(entryPath)
//                 if (isValidUtf8(fileBuffer)) {
//                     const fileContent = fileBuffer.toString('utf8')
//                     if (fileContent.trim() !== '') {
//                         allText += `${fileContent}${delimiter}`
//                     }
//                 }
//             }
//         }
//     }

//     const directories = dirs.split(',')

//     for (const directory of directories) {
//         readDirSync(directory)
//     }

//     return allText
// }

async function fetchURLSampler(
    url = 'https://www.gutenberg.org/files/100/old/shaks12.txt',
    path = 'shaks12.txt'
) {
    const fs = await import('fs')

    async function fetchAndSaveContent(url, filePath) {
        const response = await fetch(url)
        const text = await response.text()

        fs.mkdirSync('./data/datasets', { recursive: true })
        fs.writeFile(filePath, text, (err) => {
            if (err) {
                console.error('Error saving file:', err)
            } else {
                console.log('File saved successfully!')
            }
        })
        return text
    }
    const text = await fetchAndSaveContent(url, `./data/datasets/${path}`)
    console.log(text.slice(0, 300))
    return text
}

async function* CosmopediaSampler(sampleLen) {
    const CosmopediaDataset = (await import('./samplers/cosmopedia.js')).default
    const sampler = new CosmopediaDataset()
    await sampler.init()
    sampler.loadSchema([{ prompt: 'INPUT: ' }, { text: 'OUTPUT: ' }])
    while (true) {
        yield await sampler.getSample(sampleLen)
    }
}

class RandomSampler {
    constructor(sampler) {
        this.sampler = sampler
    }

    async take(config) {
        const string = await this.sampler.take(config)
        const expandedLen = config.maxSeqLen * 10
        const startIndex = Math.floor(
            Math.random() * (string.length - expandedLen)
        )
        return string.substring(startIndex, startIndex + expandedLen)
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

class CachingSampler {
    constructor(generator) {
        this.generator = generator
        this.tokens = []
    }

    async take({ tokenizer, maxSeqLen, stride } = {}) {
        while (true) {
            if (this.tokens.length >= maxSeqLen) {
                const returnTokens = this.tokens.slice(0, maxSeqLen)
                stride = Math.ceil(maxSeqLen / 2)
                this.tokens = this.tokens.slice(stride)
                return returnTokens
            }
            const sample = await this.generator.next()
            this.tokens.push(...tokenizer.encode(sample.value))
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

const samplers = {
    // stringSampler: (sampleLen, overfit, str) =>
    //     stringSampler(sampleLen, overfit, str),
    // sequentialStringSampler: (sampleLen, str) =>
    //     sequentialStringSampler(sampleLen, str),
    // directorySampler: (dir, delimiter) => directorySampler(dir, delimiter),
    // fetchURLSampler: (url, path) => fetchURLSampler(url, path),
    DirectorySampler: (directories, delimiter) =>
        new RandomSampler(new DirectorySampler(directories, delimiter)),
    CosmopediaSampler: (sampleLen) =>
        new CachingSampler(CosmopediaSampler(sampleLen)),
    MultiSampler: (samplers) => new MultiSampler(samplers)
}
export default samplers
