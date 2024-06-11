import { delay } from './utils.js'

function* stringSampler(sampleLen, overfit = 0, str = '') {
    if (overfit > 0) str = splitLines(str, overfit)
    while (true) {
        // Generate a random start index within the string's bounds
        const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
        // Yield a ${sampleLen} substring
        yield str.substring(startIndex, startIndex + sampleLen)
    }
}

function* sinWaveOscillator(min, max) {
    let i = 0
    const amplitude = (max - min) / 2 // Height from center to peak
    const offset = (max + min) / 2 // Midpoint between max and min
    while (true) {
        const value = Math.sin(i) * amplitude + offset
        yield Math.round(value)
        i += 0.1 // Increment angle to move along the sine wave
    }
}

function* sequentialStringSampler(sampleLen, overfit = 0, str) {
    let index = 0
    const oscillator = sinWaveOscillator(1, 64)
    if (overfit > 0) str = splitLines(str, overfit)
    while (true) {
        if (index + sampleLen > str.length) {
            index = 0
        }
        yield str.substring(index, index + sampleLen) // Yield a substring of length sampleLen
        index = index + oscillator.next().value // Make [x] time step over the str
    }
}

async function directorySampler(
    sampleLen,
    overfit = 0,
    dir = './',
    delimiter = '\n\n'
) {
    let allText = await directoryReader(dir, delimiter)
    return stringSampler(sampleLen, overfit, allText)
}

async function directoryReader(dir = './', delimiter = '\n\n') {
    const fs = (await import('fs')).default
    const path = (await import('path')).default

    let allText = ''

    const readDirSync = (currentPath) => {
        fs.readdirSync(currentPath, { withFileTypes: true }).forEach(
            (entry) => {
                const entryPath = path.join(currentPath, entry.name)
                if (entry.isDirectory()) {
                    readDirSync(entryPath)
                } else {
                    allText += `${fs.readFileSync(
                        entryPath,
                        'utf8'
                    )}${delimiter}`
                }
            }
        )
    }

    readDirSync(dir)

    // console.log(Array.from(new Set(allText)).sort().join(''))

    return allText
}

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

class GunSampler {
    constructor(config) {
        this.gun
        this.config = config
    }

    async init() {
        const Gun = (await import('gun')).default
        this.gun = new Gun({
            peers: ['wss://59.src.eco/gun', 'wss://95.src.eco/gun'],
            localStorage: true,
            radisk: false,
            axe: false,
            file: './data/gun'
        })
    }

    async uploadDirectory(key = 'phi', dir = './src') {
        await this.putDataset(key, await directoryReader(dir, '\n\n\n'))
    }

    async putDataset(key, object) {
        this.gun.get('src').get('datasets').get(key).put(object)
    }

    async getDataset(key) {
        let data = false
        this.gun
            .get('src')
            .get('datasets')
            .get(key)
            .once(async (node) => {
                data = node
            })
        while (!data) {
            console.log(`Retreiving [${key}] dataset...`)
            await delay(5000)
        }
        return data
    }

    async subscribeChannel(key = 'trade') {
        this.gun
            .get('src')
            .get('bullets')
            .get(key)
            .on(async (node) => {
                console.log(node)
            })
    }
}

const samplers = {
    stringSampler: (sampleLen, overfit, str) =>
        stringSampler(sampleLen, overfit, str),
    sequentialStringSampler: (sampleLen, overfit, str) =>
        sequentialStringSampler(sampleLen, overfit, str),
    directorySampler: (sampleLen, overfit, dir, delimiter) =>
        directorySampler(sampleLen, overfit, dir, delimiter),
    gunSampler: (config) => new GunSampler(config),
    fetchURLSampler: (url, path) => fetchURLSampler(url, path)
}
export default samplers
