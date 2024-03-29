import { AutoTokenizer } from '@xenova/transformers'
;(async function () {
    const tokenizer = await AutoTokenizer.from_pretrained('Xenova/t5-small')

    const { input_ids } = tokenizer('Hello world!')
    console.log(input_ids)
})()
