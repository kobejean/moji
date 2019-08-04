import TensorFlow

public struct MultiHeadAttention: Layer {
    var attention: Attention
    var wo: TimeDistributed
    @noDerivative let headCount: Int
    
    public init(size: Int, headCount: Int, causal: Bool = false, dropProbability: Double = 0.2) {
        attention = Attention(size: size, causal: causal, dropProbability: dropProbability)
        wo = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size, activation: identity))
        self.headCount = headCount
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkvProjected = attention.wqkv(input)
        let qkvSplit = splitHeads(qkvProjected, headCount: headCount)
        let qkv = Attention.QueryKeyValue.split(qkvSplit)
        let heads = attention.attend(query: qkv.query, key: qkv.key, value: qkv.value)
        return wo(joinHeads(heads, headCount: headCount))
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>, state: inout Attention.Context) -> Tensor<Float> {
        let qkvProjected = attention.wqkv(input)
        let qkvSplit = splitHeads(qkvProjected, headCount: headCount)
        let qkv = Attention.QueryKeyValue.split(qkvSplit)
        state = Attention.Context.make(
            key: state.key.concatenated(with: qkv.key, alongAxis: 1),
            value: state.value.concatenated(with: qkv.value, alongAxis: 1)
        )
        let heads = attention.attend(query: qkv.query, key: state.key, value: state.value)
        return wo(joinHeads(heads, headCount: headCount))
    }

    // MARK: Private Methods

    @differentiable(wrt: input)
    private func splitHeads(_ input: Tensor<Float>, headCount: Int) -> Tensor<Float> {
        let (batchSize, timeSteps, features) = (input.shape[0], input.shape[1], input.shape[2])
        let featuresPerHead = features / headCount
        let splitLastDim = input.reshaped(to: [batchSize, timeSteps, headCount, featuresPerHead])
        let movedToFront = splitLastDim.transposed(withPermutations: 0, 2, 1, 3)
        return movedToFront.reshaped(to: [batchSize * headCount, timeSteps, featuresPerHead])
    }

    @differentiable(wrt: input)
    private func joinHeads(_ input: Tensor<Float>, headCount: Int) -> Tensor<Float> {
        let (generalizedBatch, timeSteps, featuresPerHead) = (
            input.shape[0], input.shape[1], input.shape[2])
        let batchSize = generalizedBatch / headCount
        let features = featuresPerHead * headCount
        let splitFirstDim = input.reshaped(to: [batchSize, headCount, timeSteps, featuresPerHead])
        let movedToBack = splitFirstDim.transposed(withPermutations: 0, 2, 1, 3)
        return movedToBack.reshaped(to: [batchSize, timeSteps, features])
    }
}
