import ParquetReader from './readers/parquet.js'

export default class CosmopediaDataset extends ParquetReader {
    constructor(config) {
        super(config)
        this.dataset = 'HuggingFaceTB/cosmopedia'
        this.slices = [
            { slice: 'auto_math_text', shards: 18 },
            { slice: 'khanacademy', shards: 1 },
            { slice: 'openstax', shards: 2 },
            { slice: 'stanford', shards: 13 },
            { slice: 'stories', shards: 43 },
            { slice: 'web_samples_v1', shards: 139 },
            { slice: 'web_samples_v2', shards: 118 },
            { slice: 'wikihow', shards: 2 }
        ]
        this.schemaTemplate = config?.schema || [
            { prompt: '\nINPUT: ' },
            { text: '\nOUTPUT: ' }
        ]
    }
}
