import TensorFlow

public struct MultiHeadAttention: Layer {
    var denseQKV: TimeDistributed
    @noDerivative let attention: Attention
    var deneseO: TimeDistributed
    @noDerivative let headCount: Int
    
    public init(size: Int, headCount: Int, dropProbability: Double = 0.2) {
        denseQKV = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size * 3))
        attention = Attention(size: size, dropProbability: dropProbability)
        deneseO = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size))
        self.headCount = headCount
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkv = denseQKV(input)
        let heads = attention(splitHeads(qkv))
        return deneseO(joinHeads(heads))
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>, state: inout AttentionContext, mask: Tensor<Float>? = nil) -> Tensor<Float> {
        let qkv = denseQKV(input)
        let heads = attention(splitHeads(qkv), state: &state, mask: mask)
        return deneseO(joinHeads(heads))
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
        let (generalizedBatch, timeSteps, featuresPerHead) = (input.shape[0], input.shape[1], input.shape[2])
        let batchSize = generalizedBatch / headCount
        let features = featuresPerHead * headCount
        let splitFirstDim = input.reshaped(to: [batchSize, headCount, timeSteps, featuresPerHead])
        let movedToBack = splitFirstDim.transposed(withPermutations: 0, 2, 1, 3)
        return movedToBack.reshaped(to: [batchSize, timeSteps, features])
    }
}
