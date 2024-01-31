import * as tf from '@tensorflow/tfjs'
import { createModel } from '../src/models'

// tf.setBackend('cpu')
// tf.env().set('IS_NODE', true)

test('createModel returns a tfjs model', async () => {
  const lstmLayerSize = [128, 128]
  const sampleLen = 60
  const learningRate = 1e-2
  const charSet = Array.from(new Set(Array.from('this is training data')))
  const { length: charSetSize } = charSet
  const model = await createModel(
    lstmLayerSize,
    sampleLen,
    charSetSize,
    learningRate
  ) // Call your function to create the model

  expect(model).toBeInstanceOf(tf.LayersModel) // Assert that the returned object is a tfjs model
})

// import {
//     cryptoRandomString,
//     decrypt,
//     delay,
//     encrypt,
//     hashValue,
//     outputFilter,
//     randomString,
//     randomValueFromArray,
//     removeFirst,
//     seededValueFromArray,
//     shuffleArray
// } from '../src/common'

// test('generates a random string', () => {
//     expect(randomString(1, 'a')).toBe('a')
//     expect(randomString(3)).toHaveLength(3)
// })

// // test('generates a cryptorandom string with length of 30', () => {
// //     expect(cryptoRandomString(30)).toHaveLength(30)
// // })

// test('test encryption and decryption', () => {
//     const encrypted = encrypt('This is a test.', 'password123')
//     expect(decrypt(encrypted, 'password123')).toBe('This is a test.')
// })

// // test('get random value from array', () => {
// //     const array = [1, 2, 3, 4, 5]
// //     expect(randomValueFromArray(array)).toHaveReturned()
// // })

// test('get a deterministic value from an array', () => {
//     const array = [1, 2, 3, 4]
//     seededValueFromArray(array, '0Y')
// })

// test('wait .25 seconds', async () => {
//     await delay(250)
//     expect.anything()
// })

// test('filter output characters', () => {
//     expect(outputFilter('🍆')).toBe(false)
// })

// test('shuffle an array', () => {
//     let array = ['a', 'b', 'c', 'd', 'e']
//     shuffleArray(array)
//     expect(array).not.toBe(['a', 'b', 'c', 'd', 'e'])
// })

// test('remove the first word of a sentence', () => {
//     let phrase = 'the quick brown fox'
//     expect(removeFirst(phrase)).toBe('quick brown fox')
// })

// test('hash a string', () => {
//     expect(hashValue('test', { size: 64 }, true)).toBe('00f9e6e6ef197c2b25')
// })
