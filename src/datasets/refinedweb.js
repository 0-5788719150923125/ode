import { parseTable } from 'arrow-js-ffi'
import initWasm, { wasmMemory, readParquet } from 'parquet-wasm'
import ParquetReader from './readers/parquet.js'

// curl -X GET "https://huggingface.co/api/datasets/tiiuae/falcon-refinedweb/parquet/default/train"
// https://huggingface.co/datasets/tiiuae/falcon-refinedweb/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet

export default class RefinedWebDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'tiiuae/falcon-refinedweb'
        this.schemaTemplate = config?.schema || [{ markdown: '\n\n' }]
        this.parquetFiles = []
        this.cachedText = {
            train: '',
            validation: ''
        }
    }

    async fetchDataset() {
        const response = await fetch(
            `https://huggingface.co/api/datasets/${this.dataset}/parquet/default/train`,
            {
                method: 'GET',
                headers: {
                    Accept: 'application/json'
                }
            }
        )

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }

        const parquetFiles = await response.json()
        return parquetFiles
    }

    async fetchRandomShard(mode = 'train') {
        if (this.parquetFiles.length === 0) {
            this.parquetFiles = await this.fetchDataset()
        }
        const url = this.rng[mode].randomValueFromArray(this.parquetFiles)
        console.log('fetching dataset:', url)
        try {
            await this.streamDataIntoTable(url)
        } catch (err) {
            console.error(err)
            console.warn(
                `Failed to fetch shard (${this.dataset}) from HuggingFace! We will continue using the old one for now...`
            )
        }
        this.loadSchema(this.schemaTemplate)
    }

    async streamDataIntoTable(url) {
        const resp = await fetch(url)
        console.log('1')
        const buffer = new Uint8Array(await resp.arrayBuffer())
        console.log('2')
        const ffiTable = readParquet(buffer).intoFFI()
        console.log('3')

        this.table = parseTable(
            wasmMemory().buffer,
            ffiTable.arrayAddrs(),
            ffiTable.schemaAddr()
        )
        console.log('4')

        ffiTable.drop()
    }

    // async fillCache(mode = 'train', batchIdx = 0) {
    //     while (this.cachedText[mode].length < this.cacheSize) {
    //         const text = []

    //         let rowIdx = null
    //         for (const field of this.schema) {
    //             let column = this.table.batches[batchIdx].getChildAt(field.idx)
    //             if (rowIdx === null) {
    //                 rowIdx = this.rng[mode].randomBetween(0, column.length - 1)
    //             }
    //             const prefix = field.value
    //             const data = column.get(rowIdx)
    //             text.push(prefix + data)
    //         }
    //         this.cachedText[mode] += text.join(this.delimiter) + this.eosToken
    //     }
    // }

    // async getSample({ mode = 'train', size = 512 }) {
    //     let batchIdx = this.trainBatchIdx
    //     if (this.seed === null) batchIdx = randomBetween(0, 1)
    //     if (mode === 'validation') batchIdx = this.validationBatchIdx
    //     await this.fillCache(mode, batchIdx)
    //     const sample = this.cachedText[mode].slice(0, size)
    //     this.cachedText[mode] = this.cachedText[mode].slice(size)
    //     return sample
    // }
}

async function main() {
    const sampler = new RefinedWebDataset({
        schema: [{ text: '' }]
    })
    await sampler.init()
    for (let i = 0; i < 10; i++) {
        console.log(await sampler.getSample({ mode: 'train', size: 512 }))
        console.log('---')
        console.log('---')
        console.log('---')
    }
}

main()
