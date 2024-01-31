import '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs'

export async function createModel(
  lstmLayerSize,
  sampleLen,
  charSetSize,
  learningRate
) {
  // XXX: Create our processing model. We iterate through the array of lstmLayerSize and
  //      iteratively add an LSTM processing layer whose number of internal units match the
  //      specified value.
  const model = lstmLayerSize.reduce((mdl, lstmLayerSize, i, orig) => {
    console.log(i)
    mdl.add(
      tf.layers.lstm({
        units: lstmLayerSize,
        // XXX: For all layers except the last one, we specify that we'll be returning
        //      sequences of data. This allows us to iteratively chain individual LSTMs
        //      to one-another.
        returnSequences: i < orig.length - 1,
        // XXX: Since each LSTM layer generates a sequence of data, only the first layer
        //      needs to receive a specific input shape. Here, we initialize the inputShape
        //      [sampleLen, charSetSize]. This defines that the first layer will receive an
        //      input matrix which allows us to convert from our selected sample range into
        //      the size of our charset. The charset uses one-hot encoding, which allows us
        //      to represent each possible character in our dataset using a dedicated input
        //      neuron.
        inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
      })
    )
    // XXX: Here we use a sequential processing model for our network. This model gets passed
    //      between each iteration, and is what we add our LSTM layers to.
    return mdl
  }, tf.sequential())

  // XXX: At the output, we use a softmax function (a normalized exponential) as the final
  //      classification layer. This is common in many neural networks. It's particularly
  //      important for this example, because we use the logit probability model (which
  //      supports regression for networks with more than 2 possible outcomes of a categorically
  //      distributed dependent variable).
  model.add(
    tf.layers.dense({
      units: charSetSize,
      activation: 'softmax'
    })
  )

  // XXX: Finally, compile the model. The optimizer is used to define the backpropagation
  //      technique that should be used for training. We use the rmsProp to help tune the
  //      learning rate that we apply individually to each neuron to help learning.
  //      We use a categoricalCrossentropy loss model which is compatible with our softmax
  //      activation output.
  model.compile({
    optimizer: tf.train.rmsprop(learningRate),
    loss: 'categoricalCrossentropy'
  })

  return model
}
