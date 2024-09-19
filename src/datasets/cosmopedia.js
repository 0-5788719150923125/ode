import ParquetReader from './readers/parquet.js'
import { generatePaddedNumbers } from '../utils.js'

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

    async fetchRandomShard(mode = 'train') {
        const { slice, shards } = this.getWeightedRandomSlice(this.slices)
        const shardIndices = generatePaddedNumbers(0, shards, 5)
        const numShards = shardIndices.slice(-1)
        const allShards = shardIndices.slice(0, -1)
        const shard = this.rng[mode].randomValueFromArray(allShards)
        const path = `data/${slice}/${this.split}-${shard}-of-${numShards}.parquet`
        const url = `https://huggingface.co/datasets/${this.dataset}/resolve/main/${path}`
        console.log(
            'fetching dataset:',
            this.dataset,
            'shard:',
            `${shard}/${numShards}`,
            'slice:',
            slice
        )
        await this.streamDataIntoTable(url)
        this.loadSchema(this.schemaTemplate)
    }
}
