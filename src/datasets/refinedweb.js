import * as arrow from 'apache-arrow'
import wasmInit, { readParquet } from 'parquet-wasm'
import ParquetReader from './readers/parquet.js'

// curl -X GET "https://huggingface.co/api/datasets/tiiuae/falcon-refinedweb/parquet/default/train"
// https://huggingface.co/datasets/tiiuae/falcon-refinedweb/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet

export default class RefinedWebDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'tiiuae/falcon-refinedweb'
        this.schemaTemplate = config?.schema || [{ content: '\n\n' }]
        this.parquetFiles = []
        this.cachedText = {
            train: '',
            validation: ''
        }
        this.table = null
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

    // FFI fails on this dataset, so we use synchronous loading
    async streamDataIntoTable(url) {
        if (this.table) this.table.drop()
        const response = await fetch(url)
        this.buffer = new Uint8Array(await response.arrayBuffer())
        // Read Parquet buffer to Arrow Table
        const arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(arrowWasmTable.intoIPCStream())
    }
}

async function main() {
    const sampler = new RefinedWebDataset({
        schema: [{ content: '' }],
        seed: 42
    })
    await sampler.init()
    console.log(sampler.viewSchema())
    for (let i = 0; i < 10; i++) {
        console.log(await sampler.getSample({ mode: 'train', size: 512 }))
        console.log('---')
        console.log('---')
        console.log('---')
    }
}

main()
