import { createCanvas } from 'canvas'
import fs from 'fs'

function createTextImage(text) {
    const size = 256
    const backgroundColor = 'white'
    const textColor = 'black'
    const fontFamily = 'Arial'
    const padding = 4

    const canvas = createCanvas(size, size)
    const ctx = canvas.getContext('2d')

    ctx.fillStyle = backgroundColor
    ctx.fillRect(0, 0, size, size)

    ctx.fillStyle = textColor
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'

    let fontSize = 32
    let lineHeight = fontSize * 1.2
    let paragraphs = text.split('\n')
    let lines = []

    while (fontSize > 4) {
        ctx.font = `${fontSize}px ${fontFamily}`
        lines = []

        for (let i = 0; i < paragraphs.length; i++) {
            let words = paragraphs[i].split(' ')
            let currentLine = ''

            for (let j = 0; j < words.length; j++) {
                let word = words[j]
                let width = ctx.measureText(`${currentLine} ${word}`).width

                if (width < size - padding * 2) {
                    currentLine += `${word} `
                } else {
                    lines.push(currentLine.trim())
                    currentLine = `${word} `
                }
            }

            if (currentLine !== '') {
                lines.push(currentLine.trim())
            }
        }

        lineHeight = fontSize * 1.2
        let totalHeight = lineHeight * lines.length

        if (totalHeight <= size - padding * 2) {
            break
        }

        fontSize -= 1
    }

    let y = (size - lineHeight * lines.length) / 2

    for (const line of lines) {
        if (line === '') {
            y += lineHeight // Skip empty lines between paragraphs
        } else {
            ctx.fillText(line, padding, y)
            y += lineHeight
        }
    }

    const buffer = canvas.toBuffer('image/png')
    fs.writeFileSync('output.png', buffer)

    return getPixels(ctx, size)
}

function getPixels(ctx, size) {
    const imageData = ctx.getImageData(0, 0, size, size)
    const pixels = imageData.data
    const tokenizedData = []

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i]
        const g = pixels[i + 1]
        const b = pixels[i + 2]
        const a = pixels[i + 3]

        // Check if the pixel is text (black) or whitespace (white)
        const isText = r === 0 && g === 0 && b === 0 && a === 255
        const token = isText ? 1 : 0
        tokenizedData.push(token)
    }
    return tokenizedData
}

// Example usage
const textData = `In this updated code:

Two additional pairs of convolutional and pooling layers have been added.

The second convolutional layer has 64 filters, and the third convolutional layer has 128 filters.
Each convolutional layer is followed by a max pooling layer with a pool size of [2, 2] and strides of [2, 2], which reduces the spatial dimensions by half.


The number of units in the first dense layer has been increased to 256 to accommodate the reduced dimensionality of the flattened output.

By adding more convolutional and pooling layers, the spatial dimensions of the feature maps are progressively reduced, resulting in a smaller flattened output. This, in turn, reduces the number of parameters in the dense layers.
You can further experiment with the number of convolutional layers, filters, and dense units to find the optimal balance between model complexity and performance for your specific task.
Remember to monitor the model's performance and adjust the hyperparameters as needed based on your validation results.`
createTextImage(textData)
