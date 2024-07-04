import * as arrow from 'apache-arrow'
import wasmInit, { readParquet, Table } from 'parquet-wasm'
import {
    generatePaddedNumbers,
    randomBetween,
    randomValueFromArray,
    shuffleArray
} from '../utils.js'

if (typeof window !== 'undefined') await wasmInit()

export default class CosmopediaDataset {
    constructor(config) {
        this.dataset = 'HuggingFaceTB/cosmopedia'
        this.slices = [
            'auto_math_text',
            'khanacademy',
            'openstax',
            'stanford',
            'stories',
            'web_samples_v1',
            'web_samples_v2',
            'wikihow'
        ]
        this.slice = 'stories'
        this.split = 'train'
        this.shards = generatePaddedNumbers(0, 43, 5)
        this.delimiter = '\n\n'
        this.cacheSize = 20000
        this.cachedText = ''
        this.cycleShardInterval = 100
        this.batches = 0
    }

    async init() {
        await this.fetchRandomShard()
    }

    async fetchRandomShard() {
        const shard = randomValueFromArray(this.shards)
        console.log('fetching shard:', shard)
        const path = `data/${this.slice}/${
            this.split
        }-${shard}-of-${this.shards.slice(-1)}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        const response = await fetch(url)
        console.log('received shard:', shard)
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        this.buffer = new Uint8Array(await response.arrayBuffer())
        this.moveDataIntoTable()
        console.log('moved shard to table:', shard)
    }

    moveDataIntoTable() {
        // Read Parquet buffer to Arrow Table
        const arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())
    }

    viewSchema() {
        console.log(table.schema.toString())
    }

    loadSchema(array) {
        this.schema = []
        array.map((obj) => {
            Object.entries(obj).forEach(([key, value]) => {
                const idx = this.table.schema.fields.findIndex(
                    (field) => field.name === key
                )
                console.assert(
                    idx !== -1,
                    `the key of ${key} does not exist in this dataset`
                )
                this.schema.push({ idx, value })
            })
        })
    }

    fillCache() {
        while (this.cachedText.length < this.cacheSize) {
            let idx = 0
            let batchIdx = 0
            let rowIdx = 0
            try {
                batchIdx = randomBetween(0, this.table.batches.length)

                const text = []

                rowIdx = null
                for (const obj of this.schema) {
                    idx = obj.idx
                    const column = this.table.batches[batchIdx].getChildAt(
                        obj.idx
                    )
                    if (rowIdx === null) {
                        rowIdx = randomBetween(0, column.length)
                    }
                    const prefix = obj.value
                    text.push(prefix + column.get(rowIdx))
                }

                this.cachedText += text.join(this.delimiter) + this.delimiter
            } catch (err) {
                console.error(err)
                console.log('idx was:', idx)
                console.log('batch idx was:', batchIdx)
                console.log('row idx was:', rowIdx)
            }
        }
    }

    async getSample(size = 512) {
        this.batches++
        if (this.batches > this.cycleShardInterval) {
            this.batches = 0
            this.fetchRandomShard()
        }
        this.fillCache()
        const sample = this.cachedText.slice(0, size)
        this.cachedText = this.cachedText.slice(size, -1)
        return sample
    }
}

// async function main() {
//     const sampler = new CosmopediaDataset()
//     await sampler.init()
//     sampler.loadSchema([{ prompt: 'PROMPT: ' }, { text: 'ASSISTANT: ' }])
//     for (let i = 0; i < 10; i++) {
//         console.log(sampler.getSample())
//         console.log('---')
//         console.log('---')
//         console.log('---')
//     }
// }

// main()
