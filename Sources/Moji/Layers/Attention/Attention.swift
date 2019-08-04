import TensorFlow

public struct Attention: Layer {
    
    // MARK: QueryKeyValue Definition
    
    public struct QueryKeyValue: Differentiable {
        public var query: Tensor<Float>
        public var key: Tensor<Float>
        public var value: Tensor<Float>
        
        @differentiable(vjp: _vjpMake)
        public static func make(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>)
            -> Self {
            return Self(query: query, key: key, value: value)
        }

        public static func _vjpMake(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>)
            -> (Self, (Self.TangentVector) -> (Tensor<Float>, Tensor<Float>, Tensor<Float>)) {
            let result = Self(query: query, key: key, value: value)
            return (result, { seed in (seed.query, seed.key, seed.value) })
        }
        
        @differentiable(wrt: input, vjp: _vjpSplit)
        public static func split(_ input: Tensor<Float>) -> Self {
            let (batches, timeSteps, features) = (
                input.shape[0], input.shape[1], input.shape[2] / 3)
            let query = input.slice(
                lowerBounds: [0, 0, 0],
                upperBounds: [batches, timeSteps, features])
            let key = input.slice(
                lowerBounds: [0, 0, features],
                upperBounds: [batches, timeSteps, 2 * features])
            let value = input.slice(
                lowerBounds: [0, 0, 2 * features],
                upperBounds: [batches, timeSteps, 3 * features])
            return Self.make(query: query, key: key, value: value)
        }

        public static func _vjpSplit(_ input: Tensor<Float>)
            -> (Self, (Self.TangentVector) -> Tensor<Float>) {
            let value = Self.split(input)
            return (value, { seed in
                return Raw.concatV2([seed.query, seed.key, seed.value], axis: Tensor<Int32>(2))
            })
        }
    }
    
    // MARK: Context Definition
    
    public struct Context: Differentiable {
        public var key: Tensor<Float>
        public var value: Tensor<Float>
        
        @differentiable(wrt: (key, value), vjp: _vjpMakeContext)
        public static func make(key: Tensor<Float>, value: Tensor<Float>)
            -> Self {
                return Self(key: key, value: value)
        }

        public static func _vjpMakeContext(key: Tensor<Float>, value: Tensor<Float>)
            -> (Self, (Self.TangentVector) -> (Tensor<Float>, Tensor<Float>)) {
            let result = Self(key: key, value: value)
            return (result, { seed in (seed.key, seed.value) })
        }
    }
    
    // MARK: Properties
    var wqkv: TimeDistributed
    @noDerivative public let dropout: Dropout<Float>
    @noDerivative public let scale: Tensor<Float>
    @noDerivative public let causal: Bool
    
    // MARK: Initializer
    public init(size: Int, causal: Bool = false, dropProbability: Double = 0.2) {
        scale = Tensor(sqrt(Float(size)))
        dropout = Dropout<Float>(probability: dropProbability)
        wqkv = TimeDistributed(Dense<Float>(inputSize: size, outputSize: size * 3, activation: identity))
        self.causal = causal
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkv = QueryKeyValue.split(qkvProjected)
        return attend(query: qkv.query, key: qkv.key, value: qkv.value)
    }
    
    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Tensor<Float>, state: inout Context) -> Tensor<Float> {
        let qkvProjected = wqkv(input)
        let qkv = QueryKeyValue.split(qkvProjected)
        state = Context.make(
            key: state.key.concatenated(with: qkv.key, alongAxis: 1),
            value: state.value.concatenated(with: qkv.value, alongAxis: 1)
        )
        return attend(query: qkv.query, key: state.key, value: state.value)
    }
    
    @differentiable
    public func attend(query: Tensor<Float>, key: Tensor<Float>, value: Tensor<Float>) -> Tensor<Float>{
        let logits = batchedMatmul(query, key, adjointRight: true) / scale
        let score = maskedScore(logits)
        return batchedMatmul(dropout(score), value)
    }
    
    
    // MARK: Internal Methods
    
    @inlinable
    @differentiable(wrt: logits)
    internal func maskedScore(_ logits: Tensor<Float>) -> Tensor<Float> {
        guard causal else { return softmax(logits) }
        let (queryTimeSteps, keyTimeSteps) = (logits.shape[1], logits.shape[2])
        let ones = Tensor<Float>(ones: [1, queryTimeSteps, keyTimeSteps])
        let mask = Raw.matrixBandPart(
            ones,
            numLower: Tensor(Int32(-1)),
            numUpper: Tensor(Int32(queryTimeSteps - keyTimeSteps))
        )
        return maskedSoftmax(logits, mask: mask)
    }
}
