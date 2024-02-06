import fs from 'fs'
import Model from './src/index.js'

const lstmLayerSize = [128, 128, 128]
const sampleLen = 180
const learningRate = 1e-2
const displayLength = 180

const textContent = fs.readFileSync('./tests/shaks12.txt', 'utf8')

function* dataSampler(str, sampleLen) {
    while (true) {
        // Generate a random start index within the string's bounds
        const startIndex = Math.floor(Math.random() * (str.length - sampleLen))
        // Yield a ${sampleLen} substring from the random starting point
        yield str.substring(startIndex, startIndex + sampleLen)
    }
}

const model = new Model(lstmLayerSize, sampleLen, learningRate, displayLength)

await model.trainModel(dataSampler(textContent, sampleLen))
