import * as arrow from 'apache-arrow'
import { readParquet } from 'parquet-wasm'
import ParquetReader from './readers/parquet.js'

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
        this.arrowWasmTable = null
    }

    // FFI fails on this dataset, so we use the older, more stable method
    async streamDataIntoTable(url) {
        // if (this.arrowWasmTable !== null) this.arrowWasmTable.drop()
        const response = await fetch(url)
        this.buffer = new Uint8Array(await response.arrayBuffer())
        // Read Parquet buffer to Arrow Table
        this.arrowWasmTable = readParquet(this.buffer)
        // Convert to JS Arrow Table
        this.table = arrow.tableFromIPC(this.arrowWasmTable.intoIPCStream())
    }
}
