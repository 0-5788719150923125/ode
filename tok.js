import { AutoTokenizer } from '@xenova/transformers'
;(async function () {
    const tokenizer = await AutoTokenizer.from_pretrained(
        'Xenova/bert-base-uncased'
    )
    console.log(tokenizer)
    const { input_ids } = await tokenizer('I love transformers!')
    console.log(input_ids)
})()
