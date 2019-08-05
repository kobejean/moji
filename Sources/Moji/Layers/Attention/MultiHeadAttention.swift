import TensorFlow

public struct MultiHeadAttention: Layer {
    var layerQKV: TimeDistributed
    @noDerivative let attention: Attention
    var layerO: TimeDistributed
    @noDerivative let headCount: Int
    
    public init(size: Int, headCount: Int, causal: Bool = false, dropProbability: Double = 0.2) {
        layerQKV = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size * 3))
        attention = Attention(size: size, causal: causal, dropProbability: dropProbability)
        layerO = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size))
        self.headCount = headCount
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkv = layerQKV(input)
        let heads = attention(splitHeads(qkv))
        return layerO(joinHeads(heads))
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext) -> Tensor<Float> {
        let qkv = layerQKV(input)
        let heads = attention(splitHeads(qkv), state: &state)
        return layerO(joinHeads(heads))
    }

    
    // MARK: Private Methods

    @differentiable(wrt: input)
    private func splitHeads(_ input: Tensor<Float>) -> Tensor<Float> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let featuresPerHead = features / headCount
        let splitLastDim = input.reshaped(to: [batchSize, timeSteps, headCount, featuresPerHead])
        let movedToFront = splitLastDim.transposed(withPermutations: 0, 2, 1, 3)
        return movedToFront.reshaped(to: [batchSize * headCount, timeSteps, featuresPerHead])
    }

    @differentiable(wrt: input)
    private func joinHeads(_ input: Tensor<Float>) -> Tensor<Float> {
        let (generalizedBatch, timeSteps, featuresPerHead) = (
            input.shape[0], input.shape[1], input.shape[2])
        let batchSize = generalizedBatch / headCount
        let features = featuresPerHead * headCount
        let splitFirstDim = input.reshaped(to: [batchSize, headCount, timeSteps, featuresPerHead])
        let movedToBack = splitFirstDim.transposed(withPermutations: 0, 2, 1, 3)
        return movedToBack.reshaped(to: [batchSize, timeSteps, features])
    }
}
